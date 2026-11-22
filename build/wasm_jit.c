/*
 * Wax64 CheerpX-style Tiered Wasm-JIT (Phase 3) for jart/blink (WASIX)
 * - Tier 1: Counts m->ip executions in Blink's interpreter loop.
 * - Tier 2: Compiles hot x86-64 blocks into Wasm with an internal PC dispatch
 *           loop (supporting arbitrary intra-block loops & branches),
 *           instantiates with shared WebAssembly.Memory via browser-worker.js,
 *           and dispatches via call_indirect.
 * - Phase 3 additions:
 *   1. RIP-relative addressing, [disp32 + idx*scale], push/pop, call/ret, cmovcc/setcc
 *   2. Software TLB (g_tlb_r / g_tlb_w) for direct shared-memory i64.load/store
 *   3. 4-way set-associative JIT cache + WasmJitInvalidateRange on mmap/munmap/mprotect
 *   4. Named Wasm opcode enum (enum WasmOp)
 */

#ifdef BLINK_WASM_JIT

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "blink/machine.h"
#include "blink/x86.h"

#define WAX_JIT_SETS      1024
#define WAX_JIT_WAYS      4
#define WAX_JIT_THRESHOLD 32
#define WAX_JIT_MAX_INSNS 48
#define WAX_JIT_LOOP_CAP  65536
#define WAX_BUF_CAP       16384
#define WAX_TLB_SLOTS     256

enum WasmOp {
  WOP_BLOCK            = 0x02,
  WOP_LOOP             = 0x03,
  WOP_IF               = 0x04,
  WOP_ELSE             = 0x05,
  WOP_END              = 0x0b,
  WOP_BR               = 0x0c,
  WOP_BR_TABLE         = 0x0e,
  WOP_RETURN           = 0x0f,
  WOP_CALL             = 0x10,
  WOP_DROP             = 0x1a,
  WOP_SELECT           = 0x1b,
  WOP_LOCAL_GET        = 0x20,
  WOP_LOCAL_SET        = 0x21,
  WOP_LOCAL_TEE        = 0x22,
  WOP_I32_LOAD         = 0x28,
  WOP_I64_LOAD         = 0x29,
  WOP_I32_LOAD8_U      = 0x2d,
  WOP_I32_LOAD16_U     = 0x2f,
  WOP_I32_STORE        = 0x36,
  WOP_I64_STORE        = 0x37,
  WOP_I32_STORE8       = 0x3a,
  WOP_I32_STORE16      = 0x3b,
  WOP_I32_CONST        = 0x41,
  WOP_I64_CONST        = 0x42,
  WOP_I32_EQZ          = 0x45,
  WOP_I32_EQ           = 0x46,
  WOP_I32_NE           = 0x47,
  WOP_I32_LT_U         = 0x49,
  WOP_I32_GT_S         = 0x4a,
  WOP_I32_LE_U         = 0x4d,
  WOP_I64_EQZ          = 0x50,
  WOP_I64_EQ           = 0x51,
  WOP_I64_NE           = 0x52,
  WOP_I64_LT_U         = 0x54,
  WOP_I32_ADD          = 0x6a,
  WOP_I32_SUB          = 0x6b,
  WOP_I32_AND          = 0x71,
  WOP_I32_OR           = 0x72,
  WOP_I32_XOR          = 0x73,
  WOP_I32_SHL          = 0x74,
  WOP_I32_SHR_S        = 0x75,
  WOP_I32_SHR_U        = 0x76,
  WOP_I64_ADD          = 0x7c,
  WOP_I64_SUB          = 0x7d,
  WOP_I64_MUL          = 0x7e,
  WOP_I64_AND          = 0x83,
  WOP_I64_OR           = 0x84,
  WOP_I64_XOR          = 0x85,
  WOP_I64_SHL          = 0x86,
  WOP_I64_SHR_S        = 0x87,
  WOP_I64_SHR_U        = 0x88,
  WOP_I32_WRAP_I64     = 0xa7,
  WOP_I64_EXTEND_I32_S = 0xac,
  WOP_I64_EXTEND_I32_U = 0xad,
};

enum WasmType {
  WT_I32  = 0x7f,
  WT_I64  = 0x7e,
  WT_FUNC = 0x60,
  WT_VOID = 0x40,
};

__attribute__((import_module("env"), import_name("wax64_jit_compile")))
int wax64_jit_compile(const u8 *buf, int len);

struct WaxTlbEntry {
  u64 vpage;     /* guest virtual page (addr & ~4095ULL), ~0ULL when invalid */
  u32 host_base; /* Wasm32 linear memory pointer to page start, 0 when invalid */
  u32 pad;
};

static struct WaxTlbEntry g_tlb_r[WAX_TLB_SLOTS];
static struct WaxTlbEntry g_tlb_w[WAX_TLB_SLOTS];

static void tlb_flush_all(void) {
  for (int i = 0; i < WAX_TLB_SLOTS; ++i) {
    g_tlb_r[i].vpage = ~0ULL;
    g_tlb_r[i].host_base = 0;
    g_tlb_w[i].vpage = ~0ULL;
    g_tlb_w[i].host_base = 0;
  }
}

static inline void tlb_fill(struct WaxTlbEntry *tlb, struct Machine *m, u64 addr) {
  u64 vpage = addr & ~4095ULL;
  u8 *host = LookupAddress(m, vpage);
  if (host) {
    u32 idx = (u32)((vpage >> 12) & (WAX_TLB_SLOTS - 1));
    tlb[idx].vpage = vpage;
    tlb[idx].host_base = (u32)(uintptr_t)host;
  }
}

__attribute__((export_name("wax64_r8")))
u32 wax64_r8(struct Machine *m, u64 addr) {
  u8 v = 0;
  if (CopyFromUserRead(m, &v, addr, 1) != -1) {
    tlb_fill(g_tlb_r, m, addr);
  }
  return v;
}

__attribute__((export_name("wax64_r16")))
u32 wax64_r16(struct Machine *m, u64 addr) {
  u16 v = 0;
  if (CopyFromUserRead(m, &v, addr, 2) != -1) {
    if ((addr & 4095) <= 4094) tlb_fill(g_tlb_r, m, addr);
  }
  return v;
}

__attribute__((export_name("wax64_r32")))
u32 wax64_r32(struct Machine *m, u64 addr) {
  u32 v = 0;
  if (CopyFromUserRead(m, &v, addr, 4) != -1) {
    if ((addr & 4095) <= 4092) tlb_fill(g_tlb_r, m, addr);
  }
  return v;
}

__attribute__((export_name("wax64_r64")))
u64 wax64_r64(struct Machine *m, u64 addr) {
  u64 v = 0;
  if (CopyFromUserRead(m, &v, addr, 8) != -1) {
    if ((addr & 4095) <= 4088) tlb_fill(g_tlb_r, m, addr);
  }
  return v;
}

__attribute__((export_name("wax64_w8")))
void wax64_w8(struct Machine *m, u64 addr, u32 v) {
  u8 b = (u8)v;
  if (CopyToUserWrite(m, addr, &b, 1) != -1) {
    tlb_fill(g_tlb_w, m, addr);
  }
}

__attribute__((export_name("wax64_w16")))
void wax64_w16(struct Machine *m, u64 addr, u32 v) {
  u16 w = (u16)v;
  if (CopyToUserWrite(m, addr, &w, 2) != -1) {
    if ((addr & 4095) <= 4094) tlb_fill(g_tlb_w, m, addr);
  }
}

__attribute__((export_name("wax64_w32")))
void wax64_w32(struct Machine *m, u64 addr, u32 v) {
  if (CopyToUserWrite(m, addr, &v, 4) != -1) {
    if ((addr & 4095) <= 4092) tlb_fill(g_tlb_w, m, addr);
  }
}

__attribute__((export_name("wax64_w64")))
void wax64_w64(struct Machine *m, u64 addr, u64 v) {
  if (CopyToUserWrite(m, addr, &v, 8) != -1) {
    if ((addr & 4095) <= 4088) tlb_fill(g_tlb_w, m, addr);
  }
}

typedef void (*wax_jit_fn)(struct Machine *);

struct WaxJitSlot {
  u64 ip;
  u32 hits;
  u32 lru;
  u8 state; /* 0 = cold, 1 = compiled, 2 = unsupported */
  wax_jit_fn fn;
};

static struct WaxJitSlot g_jit_cache[WAX_JIT_SETS][WAX_JIT_WAYS];
static u32 g_jit_clock = 0;

void WasmJitReset(void) {
  memset(g_jit_cache, 0, sizeof(g_jit_cache));
  g_jit_clock = 0;
  tlb_flush_all();
}

void WasmJitInvalidateRange(u64 start, u64 len) {
  tlb_flush_all();
  if (len == 0 || start + len < start) {
    memset(g_jit_cache, 0, sizeof(g_jit_cache));
    return;
  }
  u64 lo = start > 256 ? start - 256 : 0;
  u64 hi = start + len;
  for (int s = 0; s < WAX_JIT_SETS; ++s) {
    for (int w = 0; w < WAX_JIT_WAYS; ++w) {
      struct WaxJitSlot *slot = &g_jit_cache[s][w];
      if ((slot->state != 0 || slot->hits > 0) && slot->ip >= lo && slot->ip < hi) {
        slot->ip = 0;
        slot->hits = 0;
        slot->lru = 0;
        slot->state = 0;
        slot->fn = NULL;
      }
    }
  }
}

struct WBuf {
  u8 data[WAX_BUF_CAP];
  int len;
  int err;
};

static inline void wb_u8(struct WBuf *b, u8 v) {
  if (b->len < WAX_BUF_CAP) b->data[b->len++] = v;
  else b->err = 1;
}

static void wb_uleb(struct WBuf *b, u32 v) {
  do {
    u8 byte = v & 0x7f;
    v >>= 7;
    if (v) byte |= 0x80;
    wb_u8(b, byte);
  } while (v);
}

static void wb_sleb64(struct WBuf *b, i64 v) {
  for (;;) {
    u8 byte = (u8)(v & 0x7f);
    v >>= 7;
    int done = ((v == 0 && (byte & 0x40) == 0) || (v == -1 && (byte & 0x40) != 0));
    if (!done) byte |= 0x80;
    wb_u8(b, byte);
    if (done) break;
  }
}

static void wb_sleb32(struct WBuf *b, i32 v) {
  wb_sleb64(b, (i64)v);
}

static void wb_str(struct WBuf *b, const char *s) {
  u32 n = (u32)strlen(s);
  wb_uleb(b, n);
  for (u32 i = 0; i < n; ++i) wb_u8(b, (u8)s[i]);
}

/* Wasm locals in run(m: i32) */
#define L_M        0
#define L_REG(r)   (1 + (r))
#define L_TMP      17 /* i64 */
#define L_LHS      18 /* i64 */
#define L_RHS      19 /* i64 */
#define L_ADDR     20 /* i64 */
#define L_ZF       21 /* i32 */
#define L_SF       22 /* i32 */
#define L_CF       23 /* i32 */
#define L_OF       24 /* i32 */
#define L_FDIRTY   25 /* i32 */
#define L_BUDGET   26 /* i32 */
#define L_PC       27 /* i32 */
#define L_HPTR     28 /* i32 (TLB scratch) */

#define FN_R8      0
#define FN_R16     1
#define FN_R32     2
#define FN_R64     3
#define FN_W8      4
#define FN_W16     5
#define FN_W32     6
#define FN_W64     7
#define FN_RUN     8

static void op_local_get(struct WBuf *b, u32 l) { wb_u8(b, WOP_LOCAL_GET); wb_uleb(b, l); }
static void op_local_set(struct WBuf *b, u32 l) { wb_u8(b, WOP_LOCAL_SET); wb_uleb(b, l); }
static void op_local_tee(struct WBuf *b, u32 l) { wb_u8(b, WOP_LOCAL_TEE); wb_uleb(b, l); }
static void op_i32_const(struct WBuf *b, i32 v) { wb_u8(b, WOP_I32_CONST); wb_sleb32(b, v); }
static void op_i64_const(struct WBuf *b, i64 v) { wb_u8(b, WOP_I64_CONST); wb_sleb64(b, v); }
static void op_i32_load(struct WBuf *b, u32 align, u32 off) { wb_u8(b, WOP_I32_LOAD); wb_uleb(b, align); wb_uleb(b, off); }
static void op_i64_load(struct WBuf *b, u32 align, u32 off) { wb_u8(b, WOP_I64_LOAD); wb_uleb(b, align); wb_uleb(b, off); }
static void op_i32_store(struct WBuf *b, u32 align, u32 off) { wb_u8(b, WOP_I32_STORE); wb_uleb(b, align); wb_uleb(b, off); }
static void op_i64_store(struct WBuf *b, u32 align, u32 off) { wb_u8(b, WOP_I64_STORE); wb_uleb(b, align); wb_uleb(b, off); }
static void op_call(struct WBuf *b, u32 fn) { wb_u8(b, WOP_CALL); wb_uleb(b, fn); }

static void emit_flush_state(struct WBuf *b, u32 dirty_mask) {
  u32 fl_off = (u32)offsetof(struct Machine, flags);
  u32 weg_off = (u32)offsetof(struct Machine, weg);

  for (int r = 0; r < 16; ++r) {
    if (dirty_mask & (1u << r)) {
      op_local_get(b, L_M);
      op_local_get(b, L_REG(r));
      op_i64_store(b, 3, weg_off + (u32)(r * 8));
    }
  }
  op_local_get(b, L_FDIRTY);
  wb_u8(b, WOP_IF); wb_u8(b, WT_VOID);
    op_local_get(b, L_M);
    op_local_get(b, L_M);
    op_i32_load(b, 2, fl_off);
    op_i32_const(b, ~0x8C1);
    wb_u8(b, WOP_I32_AND);
    op_local_get(b, L_CF);
    wb_u8(b, WOP_I32_OR);
    op_local_get(b, L_ZF); op_i32_const(b, 6); wb_u8(b, WOP_I32_SHL);
    wb_u8(b, WOP_I32_OR);
    op_local_get(b, L_SF); op_i32_const(b, 7); wb_u8(b, WOP_I32_SHL);
    wb_u8(b, WOP_I32_OR);
    op_local_get(b, L_OF); op_i32_const(b, 11); wb_u8(b, WOP_I32_SHL);
    wb_u8(b, WOP_I32_OR);
    op_i32_store(b, 2, fl_off);
  wb_u8(b, WOP_END);
}

static void emit_exit(struct WBuf *b, u64 next_ip, u32 dirty_mask) {
  u32 ip_off = (u32)offsetof(struct Machine, ip);
  emit_flush_state(b, dirty_mask);
  op_local_get(b, L_M);
  op_i64_const(b, (i64)next_ip);
  op_i64_store(b, 3, ip_off);
  wb_u8(b, WOP_RETURN);
}

static void emit_exit_reg(struct WBuf *b, u32 ip_local, u32 dirty_mask) {
  u32 ip_off = (u32)offsetof(struct Machine, ip);
  emit_flush_state(b, dirty_mask);
  op_local_get(b, L_M);
  op_local_get(b, ip_local);
  op_i64_store(b, 3, ip_off);
  wb_u8(b, WOP_RETURN);
}

static void emit_mem_read(struct WBuf *b, int size) {
  u32 tlb_base = (u32)(uintptr_t)g_tlb_r;
  u8 res_type = (size == 8) ? WT_I64 : WT_I32;
  u32 slow_fn = (size == 8) ? FN_R64 : (size == 4) ? FN_R32 : (size == 2) ? FN_R16 : FN_R8;

  op_local_get(b, L_ADDR);
  wb_u8(b, WOP_I32_WRAP_I64);
  op_i32_const(b, 4095);
  wb_u8(b, WOP_I32_AND);
  op_i32_const(b, 4096 - size);
  wb_u8(b, WOP_I32_LE_U);
  wb_u8(b, WOP_IF); wb_u8(b, res_type);
    op_local_get(b, L_ADDR);
    op_i64_const(b, 12);
    wb_u8(b, WOP_I64_SHR_U);
    wb_u8(b, WOP_I32_WRAP_I64);
    op_i32_const(b, WAX_TLB_SLOTS - 1);
    wb_u8(b, WOP_I32_AND);
    op_i32_const(b, 4);
    wb_u8(b, WOP_I32_SHL);
    op_i32_const(b, (i32)tlb_base);
    wb_u8(b, WOP_I32_ADD);
    op_local_tee(b, L_HPTR);

    op_i64_load(b, 3, 0);
    op_local_get(b, L_ADDR);
    op_i64_const(b, ~4095LL);
    wb_u8(b, WOP_I64_AND);
    wb_u8(b, WOP_I64_EQ);

    op_local_get(b, L_HPTR);
    op_i32_load(b, 2, 8);
    op_local_tee(b, L_HPTR);
    op_i32_const(b, 0);
    wb_u8(b, WOP_I32_NE);
    wb_u8(b, WOP_I32_AND);

    wb_u8(b, WOP_IF); wb_u8(b, res_type);
      op_local_get(b, L_HPTR);
      op_local_get(b, L_ADDR);
      wb_u8(b, WOP_I32_WRAP_I64);
      op_i32_const(b, 4095);
      wb_u8(b, WOP_I32_AND);
      wb_u8(b, WOP_I32_ADD);
      if (size == 8)      { wb_u8(b, WOP_I64_LOAD);     wb_u8(b, 0x00); wb_uleb(b, 0); }
      else if (size == 4) { wb_u8(b, WOP_I32_LOAD);     wb_u8(b, 0x00); wb_uleb(b, 0); }
      else if (size == 2) { wb_u8(b, WOP_I32_LOAD16_U); wb_u8(b, 0x00); wb_uleb(b, 0); }
      else                { wb_u8(b, WOP_I32_LOAD8_U);  wb_u8(b, 0x00); wb_uleb(b, 0); }
    wb_u8(b, WOP_ELSE);
      op_local_get(b, L_M);
      op_local_get(b, L_ADDR);
      op_call(b, slow_fn);
    wb_u8(b, WOP_END);
  wb_u8(b, WOP_ELSE);
    op_local_get(b, L_M);
    op_local_get(b, L_ADDR);
    op_call(b, slow_fn);
  wb_u8(b, WOP_END);
}

static void emit_mem_write(struct WBuf *b, int size, u32 val_i64_local) {
  u32 tlb_base = (u32)(uintptr_t)g_tlb_w;
  u32 slow_fn = (size == 8) ? FN_W64 : (size == 4) ? FN_W32 : (size == 2) ? FN_W16 : FN_W8;

  op_local_get(b, L_ADDR);
  wb_u8(b, WOP_I32_WRAP_I64);
  op_i32_const(b, 4095);
  wb_u8(b, WOP_I32_AND);
  op_i32_const(b, 4096 - size);
  wb_u8(b, WOP_I32_LE_U);
  wb_u8(b, WOP_IF); wb_u8(b, WT_VOID);
    op_local_get(b, L_ADDR);
    op_i64_const(b, 12);
    wb_u8(b, WOP_I64_SHR_U);
    wb_u8(b, WOP_I32_WRAP_I64);
    op_i32_const(b, WAX_TLB_SLOTS - 1);
    wb_u8(b, WOP_I32_AND);
    op_i32_const(b, 4);
    wb_u8(b, WOP_I32_SHL);
    op_i32_const(b, (i32)tlb_base);
    wb_u8(b, WOP_I32_ADD);
    op_local_tee(b, L_HPTR);

    op_i64_load(b, 3, 0);
    op_local_get(b, L_ADDR);
    op_i64_const(b, ~4095LL);
    wb_u8(b, WOP_I64_AND);
    wb_u8(b, WOP_I64_EQ);

    op_local_get(b, L_HPTR);
    op_i32_load(b, 2, 8);
    op_local_tee(b, L_HPTR);
    op_i32_const(b, 0);
    wb_u8(b, WOP_I32_NE);
    wb_u8(b, WOP_I32_AND);

    wb_u8(b, WOP_IF); wb_u8(b, WT_VOID);
      op_local_get(b, L_HPTR);
      op_local_get(b, L_ADDR);
      wb_u8(b, WOP_I32_WRAP_I64);
      op_i32_const(b, 4095);
      wb_u8(b, WOP_I32_AND);
      wb_u8(b, WOP_I32_ADD);
      op_local_get(b, val_i64_local);
      if (size == 8)      { wb_u8(b, WOP_I64_STORE);   wb_u8(b, 0x00); wb_uleb(b, 0); }
      else if (size == 4) { wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_STORE);   wb_u8(b, 0x00); wb_uleb(b, 0); }
      else if (size == 2) { wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_STORE16); wb_u8(b, 0x00); wb_uleb(b, 0); }
      else                { wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_STORE8);  wb_u8(b, 0x00); wb_uleb(b, 0); }
    wb_u8(b, WOP_ELSE);
      op_local_get(b, L_M);
      op_local_get(b, L_ADDR);
      op_local_get(b, val_i64_local);
      if (size < 8) wb_u8(b, WOP_I32_WRAP_I64);
      op_call(b, slow_fn);
    wb_u8(b, WOP_END);
  wb_u8(b, WOP_ELSE);
    op_local_get(b, L_M);
    op_local_get(b, L_ADDR);
    op_local_get(b, val_i64_local);
    if (size < 8) wb_u8(b, WOP_I32_WRAP_I64);
    op_call(b, slow_fn);
  wb_u8(b, WOP_END);
}

static void emit_load_reg(struct WBuf *b, int r, int is64) {
  op_local_get(b, L_REG(r));
  if (!is64) {
    op_i64_const(b, 0xffffffffLL);
    wb_u8(b, WOP_I64_AND);
  }
}

static void emit_store_reg(struct WBuf *b, int r, int is64, u32 *dirty_mask) {
  if (!is64) {
    op_i64_const(b, 0xffffffffLL);
    wb_u8(b, WOP_I64_AND);
  }
  op_local_set(b, L_REG(r));
  *dirty_mask |= (1u << r);
}

static void emit_logic_flags(struct WBuf *b, int is64) {
  if (is64) {
    op_local_get(b, L_TMP); wb_u8(b, WOP_I64_EQZ); op_local_set(b, L_ZF);
    op_local_get(b, L_TMP); op_i64_const(b, 63); wb_u8(b, WOP_I64_SHR_U); wb_u8(b, WOP_I32_WRAP_I64); op_local_set(b, L_SF);
  } else {
    op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_EQZ); op_local_set(b, L_ZF);
    op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); op_i32_const(b, 31); wb_u8(b, WOP_I32_SHR_U); op_local_set(b, L_SF);
  }
  op_i32_const(b, 0); op_local_set(b, L_CF);
  op_i32_const(b, 0); op_local_set(b, L_OF);
  op_i32_const(b, 1); op_local_set(b, L_FDIRTY);
}

static void emit_add_flags(struct WBuf *b, int is64) {
  if (is64) {
    op_local_get(b, L_TMP); wb_u8(b, WOP_I64_EQZ); op_local_set(b, L_ZF);
    op_local_get(b, L_TMP); op_i64_const(b, 63); wb_u8(b, WOP_I64_SHR_U); wb_u8(b, WOP_I32_WRAP_I64); op_local_set(b, L_SF);
    op_local_get(b, L_TMP); op_local_get(b, L_LHS); wb_u8(b, WOP_I64_LT_U); op_local_set(b, L_CF);
    op_local_get(b, L_LHS); op_local_get(b, L_TMP); wb_u8(b, WOP_I64_XOR);
    op_local_get(b, L_RHS); op_local_get(b, L_TMP); wb_u8(b, WOP_I64_XOR);
    wb_u8(b, WOP_I64_AND); op_i64_const(b, 63); wb_u8(b, WOP_I64_SHR_U); wb_u8(b, WOP_I32_WRAP_I64); op_local_set(b, L_OF);
  } else {
    op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_EQZ); op_local_set(b, L_ZF);
    op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); op_i32_const(b, 31); wb_u8(b, WOP_I32_SHR_U); op_local_set(b, L_SF);
    op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); op_local_get(b, L_LHS); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_LT_U); op_local_set(b, L_CF);
    op_local_get(b, L_LHS); wb_u8(b, WOP_I32_WRAP_I64); op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_XOR);
    op_local_get(b, L_RHS); wb_u8(b, WOP_I32_WRAP_I64); op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_XOR);
    wb_u8(b, WOP_I32_AND); op_i32_const(b, 31); wb_u8(b, WOP_I32_SHR_U); op_local_set(b, L_OF);
  }
  op_i32_const(b, 1); op_local_set(b, L_FDIRTY);
}

static void emit_sub_flags(struct WBuf *b, int is64) {
  if (is64) {
    op_local_get(b, L_TMP); wb_u8(b, WOP_I64_EQZ); op_local_set(b, L_ZF);
    op_local_get(b, L_TMP); op_i64_const(b, 63); wb_u8(b, WOP_I64_SHR_U); wb_u8(b, WOP_I32_WRAP_I64); op_local_set(b, L_SF);
    op_local_get(b, L_LHS); op_local_get(b, L_RHS); wb_u8(b, WOP_I64_LT_U); op_local_set(b, L_CF);
    op_local_get(b, L_LHS); op_local_get(b, L_RHS); wb_u8(b, WOP_I64_XOR);
    op_local_get(b, L_LHS); op_local_get(b, L_TMP); wb_u8(b, WOP_I64_XOR);
    wb_u8(b, WOP_I64_AND); op_i64_const(b, 63); wb_u8(b, WOP_I64_SHR_U); wb_u8(b, WOP_I32_WRAP_I64); op_local_set(b, L_OF);
  } else {
    op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_EQZ); op_local_set(b, L_ZF);
    op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); op_i32_const(b, 31); wb_u8(b, WOP_I32_SHR_U); op_local_set(b, L_SF);
    op_local_get(b, L_LHS); wb_u8(b, WOP_I32_WRAP_I64); op_local_get(b, L_RHS); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_LT_U); op_local_set(b, L_CF);
    op_local_get(b, L_LHS); wb_u8(b, WOP_I32_WRAP_I64); op_local_get(b, L_RHS); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_XOR);
    op_local_get(b, L_LHS); wb_u8(b, WOP_I32_WRAP_I64); op_local_get(b, L_TMP); wb_u8(b, WOP_I32_WRAP_I64); wb_u8(b, WOP_I32_XOR);
    wb_u8(b, WOP_I32_AND); op_i32_const(b, 31); wb_u8(b, WOP_I32_SHR_U); op_local_set(b, L_OF);
  }
  op_i32_const(b, 1); op_local_set(b, L_FDIRTY);
}

static void emit_jcc_cond(struct WBuf *b, int cc) {
  int inv = cc & 1;
  switch (cc >> 1) {
    case 0: op_local_get(b, L_OF); break;
    case 1: op_local_get(b, L_CF); break;
    case 2: op_local_get(b, L_ZF); break;
    case 3: op_local_get(b, L_CF); op_local_get(b, L_ZF); wb_u8(b, WOP_I32_OR); break;
    case 4: op_local_get(b, L_SF); break;
    case 6: op_local_get(b, L_SF); op_local_get(b, L_OF); wb_u8(b, WOP_I32_XOR); break;
    case 7:
      op_local_get(b, L_ZF);
      op_local_get(b, L_SF); op_local_get(b, L_OF); wb_u8(b, WOP_I32_XOR);
      wb_u8(b, WOP_I32_OR); break;
    default: op_i32_const(b, 0); break;
  }
  if (inv) wb_u8(b, WOP_I32_EQZ);
}

static inline u32 read_u32(const u8 *p) { u32 v; memcpy(&v, p, 4); return v; }
static inline i32 read_i32(const u8 *p) { i32 v; memcpy(&v, p, 4); return v; }
static inline i64 read_i64(const u8 *p) { i64 v; memcpy(&v, p, 8); return v; }

static int decode_modrm_addr(const u8 *code, int pos, int max, u8 rex, u64 start_ip, struct WBuf *b, int *out_reg) {
  if (pos >= max) return -1;
  u8 modrm = code[pos++];
  int mod = (modrm >> 6) & 3;
  int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
  int rm = (modrm & 7) | ((rex & 1) ? 8 : 0);
  *out_reg = reg;

  if (mod == 3) return -2;

  int base = rm;
  int has_sib = ((rm & 7) == 4);
  int idx = -1, scale = 0;

  if (has_sib) {
    if (pos >= max) return -1;
    u8 sib = code[pos++];
    scale = (sib >> 6) & 3;
    int idx_raw = ((sib >> 3) & 7) | ((rex & 2) ? 8 : 0);
    base = (sib & 7) | ((rex & 1) ? 8 : 0);
    if (idx_raw != 4) idx = idx_raw;
  }

  i64 disp = 0;
  int no_base = 0;
  if (mod == 0 && (base & 7) == 5) {
    if (pos + 4 > max) return -1;
    i32 d32 = read_i32(code + pos);
    pos += 4;
    if (!has_sib) {
      /* RIP-relative: effective address = next_insn_ip + disp32 */
      op_i64_const(b, (i64)(start_ip + (u64)pos + (i64)d32));
      op_local_set(b, L_ADDR);
      return pos;
    }
    /* SIB with mod == 0 and base == 5: [disp32 + idx*scale] (no base register) */
    no_base = 1;
    disp = d32;
  } else if (mod == 1) {
    if (pos >= max) return -1;
    disp = (i8)code[pos++];
  } else if (mod == 2) {
    if (pos + 4 > max) return -1;
    disp = read_i32(code + pos);
    pos += 4;
  }

  if (!no_base) {
    op_local_get(b, L_REG(base));
  } else {
    op_i64_const(b, disp);
  }
  if (idx >= 0) {
    op_local_get(b, L_REG(idx));
    if (scale > 0) {
      op_i64_const(b, scale);
      wb_u8(b, WOP_I64_SHL);
    }
    wb_u8(b, WOP_I64_ADD);
  }
  if (!no_base && disp != 0) {
    op_i64_const(b, disp);
    wb_u8(b, WOP_I64_ADD);
  }
  op_local_set(b, L_ADDR);
  return pos;
}

struct InsnSpan {
  int start;
  int end;
  int is_branch;
  int cc;        /* -2 for dynamic L_TMP exit (ret), -1 for static jmp/call, 0..15 for jcc */
  u64 target_ip;
  u64 fall_ip;
};

static int find_insn_by_ip(const struct InsnSpan *spans, int count, u64 start_ip, u64 ip) {
  for (int i = 0; i < count; ++i) {
    if (start_ip + (u64)spans[i].start == ip) return i;
  }
  return -1;
}

static wax_jit_fn CompileWasmBlock(struct Machine *m, u64 start_ip) {
  u8 code[256];
  int avail = (int)(4096 - (start_ip & 4095));
  if (avail > (int)sizeof(code)) avail = (int)sizeof(code);
  if (avail < 4 || CopyFromUserRead(m, code, start_ip, avail) == -1) {
    return NULL;
  }

  /* Pass 1: Emit each instruction into its own slice, recording IP spans */
  struct WBuf insn_bufs[WAX_JIT_MAX_INSNS];
  struct InsnSpan spans[WAX_JIT_MAX_INSNS];
  int pos = 0;
  int insns = 0;
  u32 dirty_mask = 0;

  while (pos < avail && insns < WAX_JIT_MAX_INSNS) {
    int ipos = pos;
    struct WBuf *ib = &insn_bufs[insns];
    memset(ib, 0, sizeof(*ib));
    memset(&spans[insns], 0, sizeof(spans[insns]));
    spans[insns].start = ipos;

    u8 rex = 0;
    while (pos < avail && code[pos] >= 0x40 && code[pos] <= 0x4f) {
      rex = code[pos++];
    }
    if (pos >= avail) { pos = ipos; break; }
    int is64 = (rex & 8) != 0;
    u8 op = code[pos++];

    if (op == 0x90) { /* nop */
      spans[insns].end = pos;
      insns++;
      continue;
    }

    if (op >= 0x50 && op <= 0x57) { /* push r64 */
      int r = (op - 0x50) | ((rex & 1) ? 8 : 0);
      u32 val_local = L_REG(r);
      if (r == 4) {
        op_local_get(ib, L_REG(4));
        op_local_set(ib, L_TMP);
        val_local = L_TMP;
      }
      op_local_get(ib, L_REG(4));
      op_i64_const(ib, 8);
      wb_u8(ib, WOP_I64_SUB);
      op_local_tee(ib, L_REG(4));
      op_local_set(ib, L_ADDR);
      dirty_mask |= (1u << 4);
      emit_mem_write(ib, 8, val_local);
      spans[insns].end = pos;
      insns++;
      continue;
    }

    if (op >= 0x58 && op <= 0x5f) { /* pop r64 */
      int r = (op - 0x58) | ((rex & 1) ? 8 : 0);
      op_local_get(ib, L_REG(4));
      op_local_tee(ib, L_ADDR);
      op_i64_const(ib, 8);
      wb_u8(ib, WOP_I64_ADD);
      op_local_set(ib, L_REG(4));
      dirty_mask |= (1u << 4);
      emit_mem_read(ib, 8);
      emit_store_reg(ib, r, 1, &dirty_mask);
      spans[insns].end = pos;
      insns++;
      continue;
    }

    if (op == 0xc3) { /* ret */
      op_local_get(ib, L_REG(4));
      op_local_tee(ib, L_ADDR);
      op_i64_const(ib, 8);
      wb_u8(ib, WOP_I64_ADD);
      op_local_set(ib, L_REG(4));
      dirty_mask |= (1u << 4);
      emit_mem_read(ib, 8);
      op_local_set(ib, L_TMP);
      spans[insns].end = pos;
      spans[insns].is_branch = 1;
      spans[insns].cc = -2;
      insns++;
      break;
    }

    if (op == 0xe8) { /* call rel32 */
      if (pos + 4 > avail) { pos = ipos; break; }
      i32 rel = read_i32(code + pos);
      pos += 4;
      u64 fall_ip = start_ip + (u64)pos;
      u64 target_ip = fall_ip + (i64)rel;
      op_local_get(ib, L_REG(4));
      op_i64_const(ib, 8);
      wb_u8(ib, WOP_I64_SUB);
      op_local_tee(ib, L_REG(4));
      op_local_set(ib, L_ADDR);
      dirty_mask |= (1u << 4);
      op_i64_const(ib, (i64)fall_ip);
      op_local_set(ib, L_TMP);
      emit_mem_write(ib, 8, L_TMP);
      spans[insns].end = pos;
      spans[insns].is_branch = 1;
      spans[insns].cc = -1;
      spans[insns].target_ip = target_ip;
      spans[insns].fall_ip = fall_ip;
      insns++;
      break;
    }

    if (op >= 0xb8 && op <= 0xbf) { /* mov r, imm32/64 */
      int r = (op - 0xb8) | ((rex & 1) ? 8 : 0);
      if (is64) {
        if (pos + 8 > avail) { pos = ipos; break; }
        op_i64_const(ib, read_i64(code + pos));
        pos += 8;
      } else {
        if (pos + 4 > avail) { pos = ipos; break; }
        op_i64_const(ib, (i64)(u64)read_u32(code + pos));
        pos += 4;
      }
      emit_store_reg(ib, r, is64, &dirty_mask);
      spans[insns].end = pos;
      insns++;
      continue;
    }

    if (op == 0x63 && is64 && pos < avail) { /* movsxd r64, r/m32 */
      u8 modrm = code[pos];
      if ((modrm >> 6) == 3) {
        pos++;
        int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
        int rm  = (modrm & 7) | ((rex & 1) ? 8 : 0);
        op_local_get(ib, L_REG(rm));
        wb_u8(ib, WOP_I32_WRAP_I64);
        wb_u8(ib, WOP_I64_EXTEND_I32_S);
        emit_store_reg(ib, reg, 1, &dirty_mask);
        spans[insns].end = pos;
        insns++;
        continue;
      } else {
        int reg = 0;
        int npos = decode_modrm_addr(code, pos, avail, rex, start_ip, ib, &reg);
        if (npos >= 0) {
          pos = npos;
          emit_mem_read(ib, 4);
          wb_u8(ib, WOP_I64_EXTEND_I32_S);
          emit_store_reg(ib, reg, 1, &dirty_mask);
          spans[insns].end = pos;
          insns++;
          continue;
        }
      }
      pos = ipos;
      break;
    }

    if (op == 0x01 || op == 0x03 || op == 0x09 || op == 0x0b ||
        op == 0x21 || op == 0x23 || op == 0x29 || op == 0x2b ||
        op == 0x31 || op == 0x33 || op == 0x39 || op == 0x3b ||
        op == 0x85 || op == 0x89 || op == 0x8b) {
      if (pos >= avail) { pos = ipos; break; }
      u8 modrm = code[pos];
      if ((modrm >> 6) == 3) {
        pos++;
        int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
        int rm  = (modrm & 7) | ((rex & 1) ? 8 : 0);
        int dir = (op == 0x03 || op == 0x0b || op == 0x23 || op == 0x2b || op == 0x33 || op == 0x3b || op == 0x8b);
        int dst = dir ? reg : rm;
        int src = dir ? rm : reg;
        u8 base = op & 0xfc;

        if (op == 0x89 || op == 0x8b) {
          emit_load_reg(ib, src, is64);
          emit_store_reg(ib, dst, is64, &dirty_mask);
        } else if (op == 0x85) {
          emit_load_reg(ib, rm, is64);
          emit_load_reg(ib, reg, is64);
          wb_u8(ib, WOP_I64_AND);
          op_local_set(ib, L_TMP);
          emit_logic_flags(ib, is64);
        } else {
          emit_load_reg(ib, dst, is64); op_local_tee(ib, L_LHS);
          emit_load_reg(ib, src, is64); op_local_tee(ib, L_RHS);
          if (base == 0x00) { wb_u8(ib, WOP_I64_ADD); op_local_tee(ib, L_TMP); emit_store_reg(ib, dst, is64, &dirty_mask); emit_add_flags(ib, is64); }
          else if (base == 0x08) { wb_u8(ib, WOP_I64_OR);  op_local_tee(ib, L_TMP); emit_store_reg(ib, dst, is64, &dirty_mask); emit_logic_flags(ib, is64); }
          else if (base == 0x20) { wb_u8(ib, WOP_I64_AND); op_local_tee(ib, L_TMP); emit_store_reg(ib, dst, is64, &dirty_mask); emit_logic_flags(ib, is64); }
          else if (base == 0x28) { wb_u8(ib, WOP_I64_SUB); op_local_tee(ib, L_TMP); emit_store_reg(ib, dst, is64, &dirty_mask); emit_sub_flags(ib, is64); }
          else if (base == 0x30) { wb_u8(ib, WOP_I64_XOR); op_local_tee(ib, L_TMP); emit_store_reg(ib, dst, is64, &dirty_mask); emit_logic_flags(ib, is64); }
          else if (base == 0x38) { wb_u8(ib, WOP_I64_SUB); op_local_set(ib, L_TMP); emit_sub_flags(ib, is64); }
        }
        spans[insns].end = pos;
        insns++;
        continue;
      } else if (op == 0x89 || op == 0x8b) {
        int reg = 0;
        int npos = decode_modrm_addr(code, pos, avail, rex, start_ip, ib, &reg);
        if (npos < 0) { pos = ipos; break; }
        pos = npos;
        if (op == 0x8b) {
          emit_mem_read(ib, is64 ? 8 : 4);
          if (!is64) wb_u8(ib, WOP_I64_EXTEND_I32_U);
          emit_store_reg(ib, reg, is64, &dirty_mask);
        } else {
          emit_mem_write(ib, is64 ? 8 : 4, L_REG(reg));
        }
        spans[insns].end = pos;
        insns++;
        continue;
      }
      pos = ipos;
      break;
    }

    if ((op == 0x88 || op == 0x8a) && !rex) {
      int reg = 0;
      int npos = decode_modrm_addr(code, pos, avail, rex, start_ip, ib, &reg);
      if (npos >= 0 && reg < 4) {
        pos = npos;
        if (op == 0x8a) {
          emit_mem_read(ib, 1);
          wb_u8(ib, WOP_I64_EXTEND_I32_U);
          op_local_get(ib, L_REG(reg));
          op_i64_const(ib, ~0xffLL);
          wb_u8(ib, WOP_I64_AND);
          wb_u8(ib, WOP_I64_OR);
          op_local_set(ib, L_REG(reg));
          dirty_mask |= (1u << reg);
        } else {
          emit_mem_write(ib, 1, L_REG(reg));
        }
        spans[insns].end = pos;
        insns++;
        continue;
      }
      pos = ipos;
      break;
    }

    if (op == 0x8d) { /* lea */
      int reg = 0;
      int npos = decode_modrm_addr(code, pos, avail, rex, start_ip, ib, &reg);
      if (npos >= 0) {
        pos = npos;
        op_local_get(ib, L_ADDR);
        emit_store_reg(ib, reg, is64, &dirty_mask);
        spans[insns].end = pos;
        insns++;
        continue;
      }
      pos = ipos;
      break;
    }

    if (op == 0x0f && pos < avail && (code[pos] == 0xb6 || code[pos] == 0xb7)) { /* movzx */
      u8 op2 = code[pos++];
      if (pos < avail && (code[pos] >> 6) != 3) {
        int reg = 0;
        int npos = decode_modrm_addr(code, pos, avail, rex, start_ip, ib, &reg);
        if (npos >= 0) {
          pos = npos;
          emit_mem_read(ib, op2 == 0xb6 ? 1 : 2);
          wb_u8(ib, WOP_I64_EXTEND_I32_U);
          emit_store_reg(ib, reg, is64, &dirty_mask);
          spans[insns].end = pos;
          insns++;
          continue;
        }
      }
      pos = ipos;
      break;
    }

    if (op == 0x0f && pos < avail && code[pos] >= 0x40 && code[pos] <= 0x4f) { /* cmovcc */
      u8 op2 = code[pos++];
      int cc = op2 & 0x0f;
      if ((cc & 0x0e) != 0x0a && pos < avail) {
        if ((code[pos] >> 6) == 3) {
          u8 modrm = code[pos++];
          int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
          int rm  = (modrm & 7) | ((rex & 1) ? 8 : 0);
          emit_load_reg(ib, rm, is64);
          emit_load_reg(ib, reg, is64);
          emit_jcc_cond(ib, cc);
          wb_u8(ib, WOP_SELECT);
          emit_store_reg(ib, reg, is64, &dirty_mask);
          spans[insns].end = pos;
          insns++;
          continue;
        } else {
          int reg = 0;
          int npos = decode_modrm_addr(code, pos, avail, rex, start_ip, ib, &reg);
          if (npos >= 0) {
            pos = npos;
            emit_mem_read(ib, is64 ? 8 : 4);
            if (!is64) wb_u8(ib, WOP_I64_EXTEND_I32_U);
            emit_load_reg(ib, reg, is64);
            emit_jcc_cond(ib, cc);
            wb_u8(ib, WOP_SELECT);
            emit_store_reg(ib, reg, is64, &dirty_mask);
            spans[insns].end = pos;
            insns++;
            continue;
          }
        }
      }
      pos = ipos;
      break;
    }

    if (op == 0x0f && pos < avail && code[pos] >= 0x90 && code[pos] <= 0x9f) { /* setcc */
      u8 op2 = code[pos++];
      int cc = op2 & 0x0f;
      if ((cc & 0x0e) != 0x0a && pos < avail && (code[pos] >> 6) == 3) {
        u8 modrm = code[pos++];
        int rm = (modrm & 7) | ((rex & 1) ? 8 : 0);
        if (rex || rm < 4) {
          emit_jcc_cond(ib, cc);
          wb_u8(ib, WOP_I64_EXTEND_I32_U);
          op_local_get(ib, L_REG(rm));
          op_i64_const(ib, ~0xffLL);
          wb_u8(ib, WOP_I64_AND);
          wb_u8(ib, WOP_I64_OR);
          op_local_set(ib, L_REG(rm));
          dirty_mask |= (1u << rm);
          spans[insns].end = pos;
          insns++;
          continue;
        }
      }
      pos = ipos;
      break;
    }

    if (op == 0x0f && pos < avail && code[pos] == 0xaf) { /* imul r, r */
      pos++;
      if (pos < avail && (code[pos] >> 6) == 3) {
        u8 modrm = code[pos++];
        int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
        int rm  = (modrm & 7) | ((rex & 1) ? 8 : 0);
        emit_load_reg(ib, reg, is64);
        emit_load_reg(ib, rm, is64);
        wb_u8(ib, WOP_I64_MUL);
        emit_store_reg(ib, reg, is64, &dirty_mask);
        spans[insns].end = pos;
        insns++;
        continue;
      }
      pos = ipos;
      break;
    }

    if ((op == 0x81 || op == 0x83) && pos < avail && (code[pos] >> 6) == 3) {
      u8 modrm = code[pos++];
      int subop = (modrm >> 3) & 7;
      int rm = (modrm & 7) | ((rex & 1) ? 8 : 0);
      i64 imm = 0;
      if (op == 0x83) {
        if (pos >= avail) { pos = ipos; break; }
        imm = (i8)code[pos++];
      } else {
        if (pos + 4 > avail) { pos = ipos; break; }
        imm = read_i32(code + pos);
        pos += 4;
      }
      if (subop == 0 || subop == 1 || subop == 4 || subop == 5 || subop == 6 || subop == 7) {
        emit_load_reg(ib, rm, is64); op_local_tee(ib, L_LHS);
        op_i64_const(ib, imm); op_local_tee(ib, L_RHS);
        if (subop == 0) { wb_u8(ib, WOP_I64_ADD); op_local_tee(ib, L_TMP); emit_store_reg(ib, rm, is64, &dirty_mask); emit_add_flags(ib, is64); }
        else if (subop == 1) { wb_u8(ib, WOP_I64_OR);  op_local_tee(ib, L_TMP); emit_store_reg(ib, rm, is64, &dirty_mask); emit_logic_flags(ib, is64); }
        else if (subop == 4) { wb_u8(ib, WOP_I64_AND); op_local_tee(ib, L_TMP); emit_store_reg(ib, rm, is64, &dirty_mask); emit_logic_flags(ib, is64); }
        else if (subop == 5) { wb_u8(ib, WOP_I64_SUB); op_local_tee(ib, L_TMP); emit_store_reg(ib, rm, is64, &dirty_mask); emit_sub_flags(ib, is64); }
        else if (subop == 6) { wb_u8(ib, WOP_I64_XOR); op_local_tee(ib, L_TMP); emit_store_reg(ib, rm, is64, &dirty_mask); emit_logic_flags(ib, is64); }
        else if (subop == 7) { wb_u8(ib, WOP_I64_SUB); op_local_set(ib, L_TMP); emit_sub_flags(ib, is64); }
        spans[insns].end = pos;
        insns++;
        continue;
      }
      pos = ipos;
      break;
    }

    if ((op == 0xc1 || op == 0xd1) && pos < avail && (code[pos] >> 6) == 3) {
      u8 modrm = code[pos++];
      int subop = (modrm >> 3) & 7;
      int rm = (modrm & 7) | ((rex & 1) ? 8 : 0);
      if (subop == 4 || subop == 5 || subop == 7) {
        u8 sh = 1;
        if (op == 0xc1) {
          if (pos >= avail) { pos = ipos; break; }
          sh = code[pos++] & (is64 ? 63 : 31);
        }
        emit_load_reg(ib, rm, is64);
        if (subop == 4) { op_i64_const(ib, sh); wb_u8(ib, WOP_I64_SHL); }
        else if (subop == 5) { op_i64_const(ib, sh); wb_u8(ib, WOP_I64_SHR_U); }
        else if (is64) { op_i64_const(ib, sh); wb_u8(ib, WOP_I64_SHR_S); }
        else { wb_u8(ib, WOP_I32_WRAP_I64); op_i32_const(ib, sh); wb_u8(ib, WOP_I32_SHR_S); wb_u8(ib, WOP_I64_EXTEND_I32_U); }
        op_local_tee(ib, L_TMP);
        emit_store_reg(ib, rm, is64, &dirty_mask);
        if (sh != 0) emit_logic_flags(ib, is64);
        spans[insns].end = pos;
        insns++;
        continue;
      }
      pos = ipos;
      break;
    }

    if (op == 0xff && pos < avail) {
      u8 modrm = code[pos];
      int subop = (modrm >> 3) & 7;
      if ((modrm >> 6) == 3 && (subop == 0 || subop == 1)) { /* inc / dec */
        pos++;
        int rm = (modrm & 7) | ((rex & 1) ? 8 : 0);
        emit_load_reg(ib, rm, is64);
        op_i64_const(ib, 1);
        wb_u8(ib, subop == 0 ? WOP_I64_ADD : WOP_I64_SUB);
        op_local_tee(ib, L_TMP);
        emit_store_reg(ib, rm, is64, &dirty_mask);
        if (is64) {
          op_local_get(ib, L_TMP); wb_u8(ib, WOP_I64_EQZ); op_local_set(ib, L_ZF);
          op_local_get(ib, L_TMP); op_i64_const(ib, 63); wb_u8(ib, WOP_I64_SHR_U); wb_u8(ib, WOP_I32_WRAP_I64); op_local_set(ib, L_SF);
        } else {
          op_local_get(ib, L_TMP); wb_u8(ib, WOP_I32_WRAP_I64); wb_u8(ib, WOP_I32_EQZ); op_local_set(ib, L_ZF);
          op_local_get(ib, L_TMP); wb_u8(ib, WOP_I32_WRAP_I64); op_i32_const(ib, 31); wb_u8(ib, WOP_I32_SHR_U); op_local_set(ib, L_SF);
        }
        op_i32_const(ib, 1); op_local_set(ib, L_FDIRTY);
        spans[insns].end = pos;
        insns++;
        continue;
      }
      pos = ipos;
      break;
    }

    if (op >= 0x70 && op <= 0x7f) {
      if ((op & 0x0e) == 0x0a || pos >= avail) { pos = ipos; break; }
      i8 rel = (i8)code[pos++];
      spans[insns].end = pos;
      spans[insns].is_branch = 1;
      spans[insns].cc = op & 0x0f;
      spans[insns].target_ip = start_ip + (u64)pos + (i64)rel;
      spans[insns].fall_ip = start_ip + (u64)pos;
      insns++;
      continue;
    }

    if (op == 0x0f && pos < avail && code[pos] >= 0x80 && code[pos] <= 0x8f) {
      u8 op2 = code[pos++];
      if ((op2 & 0x0e) == 0x0a || pos + 4 > avail) { pos = ipos; break; }
      i32 rel = read_i32(code + pos);
      pos += 4;
      spans[insns].end = pos;
      spans[insns].is_branch = 1;
      spans[insns].cc = op2 & 0x0f;
      spans[insns].target_ip = start_ip + (u64)pos + (i64)rel;
      spans[insns].fall_ip = start_ip + (u64)pos;
      insns++;
      continue;
    }

    if (op == 0xeb || op == 0xe9) {
      i64 rel = 0;
      if (op == 0xeb) {
        if (pos >= avail) { pos = ipos; break; }
        rel = (i8)code[pos++];
      } else {
        if (pos + 4 > avail) { pos = ipos; break; }
        rel = read_i32(code + pos);
        pos += 4;
      }
      spans[insns].end = pos;
      spans[insns].is_branch = 1;
      spans[insns].cc = -1;
      spans[insns].target_ip = start_ip + (u64)pos + rel;
      spans[insns].fall_ip = start_ip + (u64)pos;
      insns++;
      break;
    }

    pos = ipos;
    break;
  }

  if (insns < 2) return NULL;

  /* Pass 2: Assemble intra-block PC dispatch loop */
  struct WBuf body;
  memset(&body, 0, sizeof(body));

  u32 weg_off = (u32)offsetof(struct Machine, weg);
  u32 fl_off = (u32)offsetof(struct Machine, flags);
  for (int r = 0; r < 16; ++r) {
    op_local_get(&body, L_M);
    op_i64_load(&body, 3, weg_off + (u32)(r * 8));
    op_local_set(&body, L_REG(r));
  }
  op_local_get(&body, L_M); op_i32_load(&body, 2, fl_off); op_local_tee(&body, L_FDIRTY);
  op_i32_const(&body, 1); wb_u8(&body, WOP_I32_AND); op_local_set(&body, L_CF);
  op_local_get(&body, L_FDIRTY); op_i32_const(&body, 6); wb_u8(&body, WOP_I32_SHR_U); op_i32_const(&body, 1); wb_u8(&body, WOP_I32_AND); op_local_set(&body, L_ZF);
  op_local_get(&body, L_FDIRTY); op_i32_const(&body, 7); wb_u8(&body, WOP_I32_SHR_U); op_i32_const(&body, 1); wb_u8(&body, WOP_I32_AND); op_local_set(&body, L_SF);
  op_local_get(&body, L_FDIRTY); op_i32_const(&body, 11); wb_u8(&body, WOP_I32_SHR_U); op_i32_const(&body, 1); wb_u8(&body, WOP_I32_AND); op_local_set(&body, L_OF);
  op_i32_const(&body, 0); op_local_set(&body, L_FDIRTY);
  op_i32_const(&body, WAX_JIT_LOOP_CAP); op_local_set(&body, L_BUDGET);
  op_i32_const(&body, 0); op_local_set(&body, L_PC);

  /* loop (dispatch) */
  wb_u8(&body, WOP_LOOP); wb_u8(&body, WT_VOID);
    for (int i = 0; i <= insns; ++i) {
      wb_u8(&body, WOP_BLOCK); wb_u8(&body, WT_VOID);
    }
    op_local_get(&body, L_PC);
    wb_u8(&body, WOP_BR_TABLE);
    wb_uleb(&body, (u32)insns);
    for (int i = 0; i <= insns; ++i) wb_uleb(&body, (u32)i);
    wb_u8(&body, WOP_END);

    for (int i = 0; i < insns; ++i) {
      for (int k = 0; k < insn_bufs[i].len; ++k) {
        wb_u8(&body, insn_bufs[i].data[k]);
      }
      if (spans[i].is_branch) {
        if (spans[i].cc == -2) {
          emit_exit_reg(&body, L_TMP, dirty_mask);
        } else if (spans[i].cc == -1) {
          int tgt_idx = find_insn_by_ip(spans, insns, start_ip, spans[i].target_ip);
          if (tgt_idx >= 0) {
            op_local_get(&body, L_BUDGET);
            op_i32_const(&body, 1);
            wb_u8(&body, WOP_I32_SUB);
            op_local_tee(&body, L_BUDGET);
            op_i32_const(&body, 0);
            wb_u8(&body, WOP_I32_GT_S);
            wb_u8(&body, WOP_IF); wb_u8(&body, WT_VOID);
              op_i32_const(&body, tgt_idx);
              op_local_set(&body, L_PC);
              wb_u8(&body, WOP_BR); wb_uleb(&body, (u32)(insns - i + 1));
            wb_u8(&body, WOP_END);
            emit_exit(&body, spans[i].target_ip, dirty_mask);
          } else {
            emit_exit(&body, spans[i].target_ip, dirty_mask);
          }
        } else {
          int tgt_idx = find_insn_by_ip(spans, insns, start_ip, spans[i].target_ip);
          emit_jcc_cond(&body, spans[i].cc);
          wb_u8(&body, WOP_IF); wb_u8(&body, WT_VOID);
            if (tgt_idx >= 0) {
              op_local_get(&body, L_BUDGET);
              op_i32_const(&body, 1);
              wb_u8(&body, WOP_I32_SUB);
              op_local_tee(&body, L_BUDGET);
              op_i32_const(&body, 0);
              wb_u8(&body, WOP_I32_GT_S);
              wb_u8(&body, WOP_IF); wb_u8(&body, WT_VOID);
                op_i32_const(&body, tgt_idx);
                op_local_set(&body, L_PC);
                wb_u8(&body, WOP_BR); wb_uleb(&body, (u32)(insns - i + 2));
              wb_u8(&body, WOP_END);
              emit_exit(&body, spans[i].target_ip, dirty_mask);
            } else {
              emit_exit(&body, spans[i].target_ip, dirty_mask);
            }
          wb_u8(&body, WOP_END);
        }
      }
      wb_u8(&body, WOP_END);
    }

    emit_exit(&body, start_ip + (u64)spans[insns - 1].end, dirty_mask);
  wb_u8(&body, WOP_END);
  wb_u8(&body, WOP_END);
  if (body.err) return NULL;

  struct WBuf mod;
  memset(&mod, 0, sizeof(mod));
  const u8 hdr[8] = {0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00};
  for (int i = 0; i < 8; ++i) wb_u8(&mod, hdr[i]);

  struct WBuf sec;
  memset(&sec, 0, sizeof(sec));
  wb_uleb(&sec, 5);
  wb_u8(&sec, WT_FUNC); wb_u8(&sec, 2); wb_u8(&sec, WT_I32); wb_u8(&sec, WT_I64); wb_u8(&sec, 1); wb_u8(&sec, WT_I32);
  wb_u8(&sec, WT_FUNC); wb_u8(&sec, 2); wb_u8(&sec, WT_I32); wb_u8(&sec, WT_I64); wb_u8(&sec, 1); wb_u8(&sec, WT_I64);
  wb_u8(&sec, WT_FUNC); wb_u8(&sec, 3); wb_u8(&sec, WT_I32); wb_u8(&sec, WT_I64); wb_u8(&sec, WT_I32); wb_u8(&sec, 0);
  wb_u8(&sec, WT_FUNC); wb_u8(&sec, 3); wb_u8(&sec, WT_I32); wb_u8(&sec, WT_I64); wb_u8(&sec, WT_I64); wb_u8(&sec, 0);
  wb_u8(&sec, WT_FUNC); wb_u8(&sec, 1); wb_u8(&sec, WT_I32); wb_u8(&sec, 0);
  wb_u8(&mod, 1); wb_uleb(&mod, (u32)sec.len);
  for (int i = 0; i < sec.len; ++i) wb_u8(&mod, sec.data[i]);

  memset(&sec, 0, sizeof(sec));
  wb_uleb(&sec, 9);
  wb_str(&sec, "env"); wb_str(&sec, "memory"); wb_u8(&sec, 0x02); wb_u8(&sec, 0x03); wb_uleb(&sec, 1); wb_uleb(&sec, 65536);
  wb_str(&sec, "env"); wb_str(&sec, "r8");  wb_u8(&sec, 0x00); wb_uleb(&sec, 0);
  wb_str(&sec, "env"); wb_str(&sec, "r16"); wb_u8(&sec, 0x00); wb_uleb(&sec, 0);
  wb_str(&sec, "env"); wb_str(&sec, "r32"); wb_u8(&sec, 0x00); wb_uleb(&sec, 0);
  wb_str(&sec, "env"); wb_str(&sec, "r64"); wb_u8(&sec, 0x00); wb_uleb(&sec, 1);
  wb_str(&sec, "env"); wb_str(&sec, "w8");  wb_u8(&sec, 0x00); wb_uleb(&sec, 2);
  wb_str(&sec, "env"); wb_str(&sec, "w16"); wb_u8(&sec, 0x00); wb_uleb(&sec, 2);
  wb_str(&sec, "env"); wb_str(&sec, "w32"); wb_u8(&sec, 0x00); wb_uleb(&sec, 2);
  wb_str(&sec, "env"); wb_str(&sec, "w64"); wb_u8(&sec, 0x00); wb_uleb(&sec, 3);
  wb_u8(&mod, 2); wb_uleb(&mod, (u32)sec.len);
  for (int i = 0; i < sec.len; ++i) wb_u8(&mod, sec.data[i]);

  wb_u8(&mod, 3); wb_uleb(&mod, 2); wb_u8(&mod, 1); wb_u8(&mod, 4);

  memset(&sec, 0, sizeof(sec));
  wb_uleb(&sec, 1);
  wb_str(&sec, "run"); wb_u8(&sec, 0x00); wb_uleb(&sec, FN_RUN);
  wb_u8(&mod, 7); wb_uleb(&mod, (u32)sec.len);
  for (int i = 0; i < sec.len; ++i) wb_u8(&mod, sec.data[i]);

  memset(&sec, 0, sizeof(sec));
  wb_uleb(&sec, 2);
  wb_uleb(&sec, 20); wb_u8(&sec, WT_I64);
  wb_uleb(&sec, 8);  wb_u8(&sec, WT_I32); /* locals 21..28: i32 (includes L_PC, L_HPTR) */
  u32 func_size = (u32)(sec.len + body.len);

  struct WBuf code_sec;
  memset(&code_sec, 0, sizeof(code_sec));
  wb_uleb(&code_sec, 1);
  wb_uleb(&code_sec, func_size);
  for (int i = 0; i < sec.len; ++i) wb_u8(&code_sec, sec.data[i]);
  for (int i = 0; i < body.len; ++i) wb_u8(&code_sec, body.data[i]);
  wb_u8(&mod, 10);
  wb_uleb(&mod, (u32)code_sec.len);
  for (int i = 0; i < code_sec.len; ++i) wb_u8(&mod, code_sec.data[i]);

  if (mod.err) return NULL;
  int idx = wax64_jit_compile(mod.data, mod.len);
  return idx > 0 ? (wax_jit_fn)(uintptr_t)idx : NULL;
}

bool WasmJitDispatch(struct Machine *m) {
  if (m->mode.omode != XED_MODE_LONG) return false;
  u64 ip = m->ip;
  if (!ip) return false;

  static int tlb_inited = 0;
  if (!tlb_inited) {
    tlb_flush_all();
    tlb_inited = 1;
  }

  u32 set_idx = (u32)((ip ^ (ip >> 12)) & (WAX_JIT_SETS - 1));
  struct WaxJitSlot *set = g_jit_cache[set_idx];
  struct WaxJitSlot *match = NULL;
  struct WaxJitSlot *empty = NULL;
  struct WaxJitSlot *victim = &set[0];

  for (int w = 0; w < WAX_JIT_WAYS; ++w) {
    struct WaxJitSlot *s = &set[w];
    if ((s->state != 0 || s->hits > 0) && s->ip == ip) {
      match = s;
      break;
    }
    if (!empty && s->state == 0 && s->hits == 0) {
      empty = s;
    }
    if (s->lru < victim->lru) {
      victim = s;
    }
  }

  if (match) {
    match->lru = ++g_jit_clock;
    if (match->state == 1) {
      match->fn(m);
      return true;
    }
    if (match->state == 2) {
      return false;
    }
    if (++match->hits >= WAX_JIT_THRESHOLD) {
      wax_jit_fn fn = CompileWasmBlock(m, ip);
      if (fn) {
        match->fn = fn;
        match->state = 1;
        fn(m);
        return true;
      }
      match->state = 2;
    }
    return false;
  }

  struct WaxJitSlot *target = empty ? empty : victim;
  target->ip = ip;
  target->hits = 1;
  target->lru = ++g_jit_clock;
  target->state = 0;
  target->fn = NULL;
  return false;
}

#endif /* BLINK_WASM_JIT */