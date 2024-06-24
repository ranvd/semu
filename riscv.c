#include <stdio.h>

#include "common.h"
#include "device.h"
#include "riscv.h"
#include "riscv_private.h"

/* Return the string representation of an error code identifier */
static const char *vm_error_str(vm_error_t err)
{
    static const char *errors[] = {
        "NONE",
        "EXCEPTION",
        "USER",
    };
    if (err >= 0 && err < ARRAY_SIZE(errors))
        return errors[err];
    return "UNKNOWN";
}

/* Return a human-readable description for a RISC-V exception cause */
static const char *vm_exc_cause_str(uint32_t err)
{
    static const char *errors[] = {
        [0] = "Instruction address misaligned",
        [1] = "Instruction access fault",
        [2] = "Illegal instruction",
        [3] = "Breakpoint",
        [4] = "Load address misaligned",
        [5] = "Load access fault",
        [6] = "Store/AMO address misaligned",
        [7] = "Store/AMO access fault",
        [8] = "Environment call from U-mode",
        [9] = "Environment call from S-mode",
        [12] = "Instruction page fault",
        [13] = "Load page fault",
        [15] = "Store/AMO page fault",
    };
    if (err < ARRAY_SIZE(errors))
        return errors[err];
    return "[Unknown]";
}

void hart_error_report(const hart_t *hart)
{
    fprintf(stderr, "hart error %s: %s. val=%#x\n", vm_error_str(hart->error),
            vm_exc_cause_str(hart->exc_cause), hart->exc_val);
}

/* Instruction decoding */

/* clang-format off */
/* instruction decode masks */
enum {
    //               ....xxxx....xxxx....xxxx....xxxx
    FR_RD        = 0b00000000000000000000111110000000,
    FR_FUNCT3    = 0b00000000000000000111000000000000,
    FR_RS1       = 0b00000000000011111000000000000000,
    FR_RS2       = 0b00000001111100000000000000000000,
    //               ....xxxx....xxxx....xxxx....xxxx
    FI_IMM_11_0  = 0b11111111111100000000000000000000, // I-type
    //               ....xxxx....xxxx....xxxx....xxxx
    FS_IMM_4_0   = 0b00000000000000000000111110000000, // S-type
    FS_IMM_11_5  = 0b11111110000000000000000000000000,
    //               ....xxxx....xxxx....xxxx....xxxx
    FB_IMM_11    = 0b00000000000000000000000010000000, // B-type
    FB_IMM_4_1   = 0b00000000000000000000111100000000,
    FB_IMM_10_5  = 0b01111110000000000000000000000000,
    FB_IMM_12    = 0b10000000000000000000000000000000,
    //               ....xxxx....xxxx....xxxx....xxxx
    FU_IMM_31_12 = 0b11111111111111111111000000000000, // U-type
    //               ....xxxx....xxxx....xxxx....xxxx
    FJ_IMM_19_12 = 0b00000000000011111111000000000000, // J-type
    FJ_IMM_11    = 0b00000000000100000000000000000000,
    FJ_IMM_10_1  = 0b01111111111000000000000000000000,
    FJ_IMM_20    = 0b10000000000000000000000000000000,
    //               ....xxxx....xxxx....xxxx....xxxx
};
/* clang-format on */

/* decode U-type instruction immediate */
static inline uint32_t decode_u(uint32_t insn)
{
    return insn & FU_IMM_31_12;
}

/* decode I-type instruction immediate */
static inline uint32_t decode_i(uint32_t insn)
{
    return ((int32_t) (insn & FI_IMM_11_0)) >> 20;
}

static inline uint32_t decode_j(uint32_t insn)
{
    uint32_t dst = 0;
    dst |= (insn & FJ_IMM_20);
    dst |= (insn & FJ_IMM_19_12) << 11;
    dst |= (insn & FJ_IMM_11) << 2;
    dst |= (insn & FJ_IMM_10_1) >> 9;
    /* NOTE: shifted to 2nd least significant bit */
    return ((int32_t) dst) >> 11;
}

/* decode B-type instruction immediate */
static inline uint32_t decode_b(uint32_t insn)
{
    uint32_t dst = 0;
    dst |= (insn & FB_IMM_12);
    dst |= (insn & FB_IMM_11) << 23;
    dst |= (insn & FB_IMM_10_5) >> 1;
    dst |= (insn & FB_IMM_4_1) << 12;
    /* NOTE: shifted to 2nd least significant bit */
    return ((int32_t) dst) >> 19;
}

/* decode S-type instruction immediate */
static inline uint32_t decode_s(uint32_t insn)
{
    uint32_t dst = 0;
    dst |= (insn & FS_IMM_11_5);
    dst |= (insn & FS_IMM_4_0) << 13;
    return ((int32_t) dst) >> 20;
}

static inline uint16_t decode_i_unsigned(uint32_t insn)
{
    return insn >> 20;
}

/* decode rd field */
static inline uint8_t decode_rd(uint32_t insn)
{
    return (insn & FR_RD) >> 7;
}

/* decode rs1 field */
static inline uint8_t decode_rs1(uint32_t insn)
{
    return (insn & FR_RS1) >> 15;
}

/* decode rs2 field */
static inline uint8_t decode_rs2(uint32_t insn)
{
    return (insn & FR_RS2) >> 20;
}

/* decoded funct3 field */
static inline uint8_t decode_func3(uint32_t insn)
{
    return (insn & FR_FUNCT3) >> 12;
}

/* decoded funct5 field */
static inline uint8_t decode_func5(uint32_t insn)
{
    return insn >> 27;
}

static inline uint32_t read_rs1(const hart_t *hart, uint32_t insn)
{
    return hart->x_regs[decode_rs1(insn)];
}

static inline uint32_t read_rs2(const hart_t *hart, uint32_t insn)
{
    return hart->x_regs[decode_rs2(insn)];
}

/* virtual addressing */

static void mmu_invalidate(hart_t *hart)
{
    hart->cache_fetch.n_pages = 0xFFFFFFFF;
}

/* Pre-verify the root page table to minimize page table access during
 * translation time.
 */
static void mmu_set(hart_t *hart, uint32_t satp)
{
    mmu_invalidate(hart);
    if (satp >> 31) {
        uint32_t *page_table = hart->mem_page_table(hart, satp & MASK(22));
        if (!page_table)
            return;
        hart->page_table = page_table;
        satp &= ~(MASK(9) << 22);
    } else {
        hart->page_table = NULL;
        satp = 0;
    }
    hart->satp = satp;
}

#define PTE_ITER(page_table, vpn, additional_checks)    \
    *pte = &(page_table)[vpn];                          \
    switch ((**pte) & MASK(4)) {                        \
    case 0b0001:                                        \
        break; /* pointer to next level */              \
    case 0b0011:                                        \
    case 0b0111:                                        \
    case 0b1001:                                        \
    case 0b1011:                                        \
    case 0b1111:                                        \
        *ppn = (**pte) >> 10;                           \
        additional_checks return true; /* leaf entry */ \
    case 0b0101:                                        \
    case 0b1101:                                        \
    default:                                            \
        *pte = NULL;                                    \
        return true; /* not valid */                    \
    }

/* Assume hart->page_table is set.
 *
 * If there is an error fetching a page table, return false.
 * Otherwise return true and:
 *   - in case of valid leaf: set *pte and *ppn
 *   - none found (page fault): set *pte to NULL
 */
static bool mmu_lookup(const hart_t *hart,
                       uint32_t vpn,
                       uint32_t **pte,
                       uint32_t *ppn)
{
    PTE_ITER(hart->page_table, vpn >> 10,
             if (unlikely((*ppn) & MASK(10))) /* misaligned superpage */
                 *pte = NULL;
             else *ppn |= vpn & MASK(10);)
    uint32_t *page_table = hart->mem_page_table(hart, (**pte) >> 10);
    if (!page_table)
        return false;
    PTE_ITER(page_table, vpn & MASK(10), )
    *pte = NULL;
    return true;
}

static void mmu_translate(hart_t *hart,
                          uint32_t *addr,
                          const uint32_t access_bits,
                          const uint32_t set_bits,
                          const bool skip_privilege_test,
                          const uint8_t fault,
                          const uint8_t pfault)
{
    /* NOTE: save virtual address, for physical accesses, to set exception. */
    hart->exc_val = *addr;
    if (!hart->page_table)
        return;

    uint32_t *pte_ref;
    uint32_t ppn;
    bool ok = mmu_lookup(hart, (*addr) >> RV_PAGE_SHIFT, &pte_ref, &ppn);
    if (unlikely(!ok)) {
        hart_set_exception(hart, fault, *addr);
        return;
    }

    uint32_t pte;
    if (!(pte_ref /* PTE lookup was successful */ &&
          !(ppn >> 20) /* PPN is valid */ &&
          (pte = *pte_ref, pte & access_bits) /* access type is allowed */ &&
          (!(pte & (1 << 4)) == hart->s_mode ||
           skip_privilege_test) /* privilege matches */
          )) {
        hart_set_exception(hart, pfault, *addr);
        return;
    }

    uint32_t new_pte = pte | set_bits;
    if (new_pte != pte)
        *pte_ref = new_pte;

    *addr = ((*addr) & MASK(RV_PAGE_SHIFT)) | (ppn << RV_PAGE_SHIFT);
}

static void mmu_fence(hart_t *hart, uint32_t insn UNUSED)
{
    mmu_invalidate(hart);
}

static void mmu_fetch(hart_t *hart, uint32_t addr, uint32_t *value)
{
    uint32_t vpn = addr >> RV_PAGE_SHIFT;
    if (unlikely(vpn != hart->cache_fetch.n_pages)) {
        mmu_translate(hart, &addr, (1 << 3), (1 << 6), false, RV_EXC_FETCH_FAULT,
                      RV_EXC_FETCH_PFAULT);
        if (hart->error)
            return;
        uint32_t *page_addr;
        hart->mem_fetch(hart, addr >> RV_PAGE_SHIFT, &page_addr);
        if (hart->error)
            return;
        hart->cache_fetch.n_pages = vpn;
        hart->cache_fetch.page_addr = page_addr;
    }
    *value = hart->cache_fetch.page_addr[(addr >> 2) & MASK(RV_PAGE_SHIFT - 2)];
}

static void mmu_load(hart_t *hart,
                     uint32_t addr,
                     uint8_t width,
                     uint32_t *value,
                     bool reserved)
{
    mmu_translate(hart, &addr, (1 << 1) | (hart->sstatus_mxr ? (1 << 3) : 0),
                  (1 << 6), hart->sstatus_sum && hart->s_mode, RV_EXC_LOAD_FAULT,
                  RV_EXC_LOAD_PFAULT);
    if (hart->error)
        return;
    hart->mem_load(hart, addr, width, value);
    if (hart->error)
        return;

    if (unlikely(reserved)) {
        hart->lr_reservation = addr | 1;
        hart->lr_val = *value;
    }
}

static bool mmu_store(hart_t *hart,
                      uint32_t addr,
                      uint8_t width,
                      uint32_t value,
                      bool cond)
{
    mmu_translate(hart, &addr, (1 << 2), (1 << 6) | (1 << 7),
                  hart->sstatus_sum && hart->s_mode, RV_EXC_STORE_FAULT,
                  RV_EXC_STORE_PFAULT);
    if (hart->error)
        return false;

    if (unlikely(cond)) {
        uint32_t cas_value;
        hart->mem_load(hart, addr, width, &cas_value);
        if ((hart->lr_reservation != (addr | 1)) || hart->lr_val != cas_value)
            return false;

        hart->lr_reservation = 0;
    } else {
        if (unlikely(hart->lr_reservation & 1) &&
            (hart->lr_reservation & ~3) == (addr & ~3))
            hart->lr_reservation = 0;
    }
    hart->mem_store(hart, addr, width, value);
    return true;
}

/* exceptions, traps, interrupts */

void hart_set_exception(hart_t *hart, uint32_t cause, uint32_t val)
{
    hart->error = ERR_EXCEPTION;
    hart->exc_cause = cause;
    hart->exc_val = val;
}

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int counter = 0;
void hart_trap(hart_t *hart)
{
    /* Fill exception fields */
    hart->scause = hart->exc_cause;
    hart->stval = hart->exc_val;

    /* Save to stack */
    hart->sstatus_spie = hart->sstatus_sie;
    hart->sstatus_spp = hart->s_mode;
    hart->sepc = hart->current_pc;

    /* Set */
    hart->sstatus_sie = false;
    mmu_invalidate(hart);
    hart->s_mode = true;
    hart->pc = hart->stvec_addr;
    if (hart->stvec_vectored)
        hart->pc += (hart->scause & MASK(31)) * 4;

    hart->error = ERR_NONE;
}

static void op_sret(hart_t *hart)
{
    /* Restore from stack */
    hart->pc = hart->sepc;
    mmu_invalidate(hart);
    hart->s_mode = hart->sstatus_spp;
    hart->sstatus_sie = hart->sstatus_spie;

    /* Reset stack */
    hart->sstatus_spp = false;
    hart->sstatus_spie = true;
}

static void op_privileged(hart_t *hart, uint32_t insn)
{
    if ((insn >> 25) == 0b0001001 /* PRIV: SFENCE_VMA */) {
        mmu_fence(hart, insn);
        return;
    }
    if (insn & ((MASK(5) << 7) | (MASK(5) << 15))) {
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        return;
    }
    switch (decode_i_unsigned(insn)) {
    case 0b000000000001: /* PRIV_EBREAK */
        hart_set_exception(hart, RV_EXC_BREAKPOINT, hart->current_pc);
        break;
    case 0b000000000000: /* PRIV_ECALL */
        hart_set_exception(hart, hart->s_mode ? RV_EXC_ECALL_S : RV_EXC_ECALL_U, 0);
        break;
    case 0b000100000010: /* PRIV_SRET */
        op_sret(hart);
        break;
    case 0b000100000101: /* PRIV_WFI */
                         /* TODO: Implement this */
        break;
    default:
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        break;
    }
}

/* CSR instructions */

static inline void set_dest(hart_t *hart, uint32_t insn, uint32_t x)
{
    uint8_t rd = decode_rd(insn);
    if (rd)
        hart->x_regs[rd] = x;
}

/* clang-format off */
#define SIE_MASK (RV_INT_SEI_BIT | RV_INT_STI_BIT | RV_INT_SSI_BIT)
#define SIP_MASK (0              | 0              | RV_INT_SSI_BIT)
/* clang-format on */

static void csr_read(hart_t *hart, uint16_t addr, uint32_t *value)
{
    switch (addr) {
    case RV_CSR_TIME:
        *value = hart->time;
        return;
    case RV_CSR_TIMEH:
        *value = hart->time >> 32;
        return;
    case RV_CSR_INSTRET:
        *value = hart->instret;
        return;
    case RV_CSR_INSTRETH:
        *value = hart->instret >> 32;
        return;
    default:
        break;
    }

    if (!hart->s_mode) {
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        return;
    }

    switch (addr) {
    case RV_CSR_SSTATUS:
        *value = 0;
        hart->sstatus_sie && (*value |= 1 << (1));
        hart->sstatus_spie && (*value |= 1 << (5));
        hart->sstatus_spp && (*value |= 1 << (8));
        hart->sstatus_sum && (*value |= 1 << (18));
        hart->sstatus_mxr && (*value |= 1 << (19));
        break;
    case RV_CSR_SIE:
        *value = hart->sie;
        break;
    case RV_CSR_SIP:
        *value = hart->sip;
        break;
    case RV_CSR_STVEC:
        *value = 0;
        *value = hart->stvec_addr;
        hart->stvec_vectored && (*value |= 1 << (0));
        break;
    case RV_CSR_SATP:
        *value = hart->satp;
        break;
    case RV_CSR_SCOUNTEREN:
        *value = hart->scounteren;
        break;
    case RV_CSR_SSCRATCH:
        *value = hart->sscratch;
        break;
    case RV_CSR_SEPC:
        *value = hart->sepc;
        break;
    case RV_CSR_SCAUSE:
        *value = hart->scause;
        break;
    case RV_CSR_STVAL:
        *value = hart->stval;
        break;
    default:
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
    }
}

static void csr_write(hart_t *hart, uint16_t addr, uint32_t value)
{
    if (!hart->s_mode) {
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        return;
    }

    switch (addr) {
    case RV_CSR_SSTATUS:
        hart->sstatus_sie = (value & (1 << (1))) != 0;
        hart->sstatus_spie = (value & (1 << (5))) != 0;
        hart->sstatus_spp = (value & (1 << (8))) != 0;
        hart->sstatus_sum = (value & (1 << (18))) != 0;
        hart->sstatus_mxr = (value & (1 << (19))) != 0;
        break;
    case RV_CSR_SIE:
        value &= SIE_MASK;
        hart->sie = value;
        break;
    case RV_CSR_SIP:
        value &= SIP_MASK;
        value |= hart->sip & ~SIP_MASK;
        hart->sip = value;
        break;
    case RV_CSR_STVEC:
        hart->stvec_addr = value;
        hart->stvec_addr &= ~0b11;
        hart->stvec_vectored = (value & (1 << (0))) != 0;
        break;
    case RV_CSR_SATP:
        mmu_set(hart, value);
        break;
    case RV_CSR_SCOUNTEREN:
        hart->scounteren = value;
        break;
    case RV_CSR_SSCRATCH:
        hart->sscratch = value;
        break;
    case RV_CSR_SEPC:
        hart->sepc = value;
        break;
    case RV_CSR_SCAUSE:
        hart->scause = value;
        break;
    case RV_CSR_STVAL:
        hart->stval = value;
        break;
    default:
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
    }
}

static void op_csr_rw(hart_t *hart, uint32_t insn, uint16_t csr, uint32_t wvalue)
{
    if (decode_rd(insn)) {
        uint32_t value;
        csr_read(hart, csr, &value);
        if (unlikely(hart->error))
            return;
        set_dest(hart, insn, value);
    }
    csr_write(hart, csr, wvalue);
}

static void op_csr_cs(hart_t *hart,
                      uint32_t insn,
                      uint16_t csr,
                      uint32_t setmask,
                      uint32_t clearmask)
{
    uint32_t value;
    csr_read(hart, csr, &value);
    if (unlikely(hart->error))
        return;
    set_dest(hart, insn, value);
    if (decode_rs1(insn))
        csr_write(hart, csr, (value & ~clearmask) | setmask);
}

static void op_system(hart_t *hart, uint32_t insn)
{
    switch (decode_func3(insn)) {
    /* CSR */
    case 0b001: /* CSRRW */
        op_csr_rw(hart, insn, decode_i_unsigned(insn), read_rs1(hart, insn));
        break;
    case 0b101: /* CSRRWI */
        op_csr_rw(hart, insn, decode_i_unsigned(insn), decode_rs1(insn));
        break;
    case 0b010: /* CSRRS */
        op_csr_cs(hart, insn, decode_i_unsigned(insn), read_rs1(hart, insn), 0);
        break;
    case 0b110: /* CSRRSI */
        op_csr_cs(hart, insn, decode_i_unsigned(insn), decode_rs1(insn), 0);
        break;
    case 0b011: /* CSRRC */
        op_csr_cs(hart, insn, decode_i_unsigned(insn), 0, read_rs1(hart, insn));
        break;
    case 0b111: /* CSRRCI */
        op_csr_cs(hart, insn, decode_i_unsigned(insn), 0, decode_rs1(insn));
        break;

    /* privileged instruction */
    case 0b000: /* SYS_PRIV */
        op_privileged(hart, insn);
        break;

    default:
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        return;
    }
}

/* Unprivileged instructions */

static uint32_t op_mul(uint32_t insn, uint32_t a, uint32_t b)
{
    /* TODO: Test ifunc7 zeros */
    switch (decode_func3(insn)) {
    case 0b000: { /* MUL */
        const int64_t _a = (int32_t) a;
        const int64_t _b = (int32_t) b;
        return ((uint64_t) (_a * _b)) & ((1ULL << 32) - 1);
    }
    case 0b001: { /* MULH */
        const int64_t _a = (int32_t) a;
        const int64_t _b = (int32_t) b;
        return ((uint64_t) (_a * _b)) >> 32;
    }
    case 0b010: { /* MULHSU */
        const int64_t _a = (int32_t) a;
        const uint64_t _b = b;
        return ((uint64_t) (_a * _b)) >> 32;
    }
    case 0b011: /* MULHU */
        return (uint32_t) ((((uint64_t) a) * ((uint64_t) b)) >> 32);
    case 0b100: /* DIV */
        return b ? (a == 0x80000000 && (int32_t) b == -1)
                       ? 0x80000000
                       : (uint32_t) (((int32_t) a) / ((int32_t) b))
                 : 0xFFFFFFFF;
    case 0b101: /* DIVU */
        return b ? (a / b) : 0xFFFFFFFF;
    case 0b110: /* REM */
        return b ? (a == 0x80000000 && (int32_t) b == -1)
                       ? 0
                       : (uint32_t) (((int32_t) a) % ((int32_t) b))
                 : a;
    case 0b111: /* REMU */
        return b ? (a % b) : a;
    }
    __builtin_unreachable();
}

#define NEG_BIT (insn & (1 << 30))
static uint32_t op_rv32i(uint32_t insn, bool is_reg, uint32_t a, uint32_t b)
{
    /* TODO: Test ifunc7 zeros */
    switch (decode_func3(insn)) {
    case 0b000: /* IFUNC_ADD */
        return a + ((is_reg && NEG_BIT) ? -b : b);
    case 0b010: /* IFUNC_SLT */
        return ((int32_t) a) < ((int32_t) b);
    case 0b011: /* IFUNC_SLTU */
        return a < b;
    case 0b100: /* IFUNC_XOR */
        return a ^ b;
    case 0b110: /* IFUNC_OR */
        return a | b;
    case 0b111: /* IFUNC_AND */
        return a & b;
    case 0b001: /* IFUNC_SLL */
        return a << (b & MASK(5));
    case 0b101: /* IFUNC_SRL */
        return NEG_BIT ? (uint32_t) (((int32_t) a) >> (b & MASK(5))) /* SRA */
                       : a >> (b & MASK(5)) /* SRL */;
    }
    __builtin_unreachable();
}
#undef NEG_BIT

static bool op_jmp(hart_t *hart, uint32_t insn, uint32_t a, uint32_t b)
{
    switch (decode_func3(insn)) {
    case 0b000: /* BFUNC_BEQ */
        return a == b;
    case 0b001: /* BFUNC_BNE */
        return a != b;
    case 0b110: /* BFUNC_BLTU */
        return a < b;
    case 0b111: /* BFUNC_BGEU */
        return a >= b;
    case 0b100: /* BFUNC_BLT */
        return ((int32_t) a) < ((int32_t) b);
    case 0b101: /* BFUNC_BGE */
        return ((int32_t) a) >= ((int32_t) b);
    }
    hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
    return false;
}

static void do_jump(hart_t *hart, uint32_t addr)
{
    if (unlikely(addr & 0b11))
        hart_set_exception(hart, RV_EXC_PC_MISALIGN, addr);
    else
        hart->pc = addr;
}

static void op_jump_link(hart_t *hart, uint32_t insn, uint32_t addr)
{
    if (unlikely(addr & 0b11)) {
        hart_set_exception(hart, RV_EXC_PC_MISALIGN, addr);
    } else {
        set_dest(hart, insn, hart->pc);
        hart->pc = addr;
    }
}

#define AMO_OP(STORED_EXPR)                                   \
    do {                                                      \
        value2 = read_rs2(hart, insn);                          \
        mmu_load(hart, addr, RV_MEM_LW, &value, false);         \
        if (hart->error)                                        \
            return;                                           \
        set_dest(hart, insn, value);                            \
        mmu_store(hart, addr, RV_MEM_SW, (STORED_EXPR), false); \
    } while (0)

static void op_amo(hart_t *hart, uint32_t insn)
{
    if (unlikely(decode_func3(insn) != 0b010 /* amo.w */))
        return hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
    uint32_t addr = read_rs1(hart, insn);
    uint32_t value, value2;
    switch (decode_func5(insn)) {
    case 0b00010: /* AMO_LR */
        if (addr & 0b11)
            return hart_set_exception(hart, RV_EXC_LOAD_MISALIGN, addr);
        if (decode_rs2(insn))
            return hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        mmu_load(hart, addr, RV_MEM_LW, &value, true);
        if (hart->error)
            return;
        set_dest(hart, insn, value);
        break;
    case 0b00011: /* AMO_SC */
        if (addr & 0b11)
            return hart_set_exception(hart, RV_EXC_STORE_MISALIGN, addr);
        bool ok = mmu_store(hart, addr, RV_MEM_SW, read_rs2(hart, insn), true);
        if (hart->error)
            return;
        set_dest(hart, insn, ok ? 0 : 1);
        break;

    case 0b00001: /* AMOSWAP */
        AMO_OP(value2);
        break;
    case 0b00000: /* AMOADD */
        AMO_OP(value + value2);
        break;
    case 0b00100: /* AMOXOR */
        AMO_OP(value ^ value2);
        break;
    case 0b01100: /* AMOAND */
        AMO_OP(value & value2);
        break;
    case 0b01000: /* AMOOR */
        AMO_OP(value | value2);
        break;
    case 0b10000: /* AMOMIN */
        AMO_OP(((int32_t) value) < ((int32_t) value2) ? value : value2);
        break;
    case 0b10100: /* AMOMAX */
        AMO_OP(((int32_t) value) > ((int32_t) value2) ? value : value2);
        break;
    case 0b11000: /* AMOMINU */
        AMO_OP(value < value2 ? value : value2);
        break;
    case 0b11100: /* AMOMAXU */
        AMO_OP(value > value2 ? value : value2);
        break;
    default:
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        return;
    }
}

void hart_init(hart_t *hart)
{
    mmu_invalidate(hart);
}

#define PRIV(x) ((emu_state_t *) x->priv)
void hart_step(hart_t *hart)
{
    if (hart->hsm_status != SBI_HSM_STATE_STARTED)
        return;

    if (unlikely(hart->error))
        return;

    hart->current_pc = hart->pc;
    if ((hart->sstatus_sie || !hart->s_mode) && (hart->sip & hart->sie)) {
        uint32_t applicable = (hart->sip & hart->sie);
        uint8_t idx = __builtin_ffs(applicable) - 1;
        if (idx == 1) {
            emu_state_t *data = PRIV(hart);
            data->clint.msip[hart->mhartid] = 0;
        }
        hart->exc_cause = (1U << 31) | idx;
        hart->stval = 0;
        hart_trap(hart);
    }

    uint32_t insn;
    mmu_fetch(hart, hart->pc, &insn);
    if (unlikely(hart->error))
        return;

    hart->pc += 4;
    /* Assume no integer overflow */
    hart->instret++;

    uint32_t insn_opcode = insn & MASK(7), value;
    switch (insn_opcode) {
    case RV32_OP_IMM:
        set_dest(hart, insn,
                 op_rv32i(insn, false, read_rs1(hart, insn), decode_i(insn)));
        break;
    case RV32_OP:
        if (!(insn & (1 << 25)))
            set_dest(
                hart, insn,
                op_rv32i(insn, true, read_rs1(hart, insn), read_rs2(hart, insn)));
        else
            set_dest(hart, insn,
                     op_mul(insn, read_rs1(hart, insn), read_rs2(hart, insn)));
        break;
    case RV32_LUI:
        set_dest(hart, insn, decode_u(insn));
        break;
    case RV32_AUIPC:
        set_dest(hart, insn, decode_u(insn) + hart->current_pc);
        break;
    case RV32_JAL:
        op_jump_link(hart, insn, decode_j(insn) + hart->current_pc);
        break;
    case RV32_JALR:
        op_jump_link(hart, insn, (decode_i(insn) + read_rs1(hart, insn)) & ~1);
        break;
    case RV32_BRANCH:
        if (op_jmp(hart, insn, read_rs1(hart, insn), read_rs2(hart, insn)))
            do_jump(hart, decode_b(insn) + hart->current_pc);
        break;
    case RV32_LOAD:
        mmu_load(hart, read_rs1(hart, insn) + decode_i(insn), decode_func3(insn),
                 &value, false);
        if (unlikely(hart->error))
            return;
        set_dest(hart, insn, value);
        break;
    case RV32_STORE:
        mmu_store(hart, read_rs1(hart, insn) + decode_s(insn), decode_func3(insn),
                  read_rs2(hart, insn), false);
        if (unlikely(hart->error))
            return;
        break;
    case RV32_MISC_MEM:
        switch (decode_func3(insn)) {
        case 0b000: /* MM_FENCE */
        case 0b001: /* MM_FENCE_I */
                    /* TODO: implement for multi-threading */
            break;
        default:
            hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
            break;
        }
        break;
    case RV32_AMO:
        op_amo(hart, insn);
        break;
    case RV32_SYSTEM:
        op_system(hart, insn);
        break;
    default:
        hart_set_exception(hart, RV_EXC_ILLEGAL_INSN, 0);
        break;
    }
}
