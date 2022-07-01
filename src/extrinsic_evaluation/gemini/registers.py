
"""
The Design of Instruction Decoder

The reference come from :
1. Intel® 64 and IA-32 Architectures Software Developer’s Manual: https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
2. x64_cheatsheet: https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf
3. ORACLE x86 assembly language Reference Manual: https://docs.oracle.com/cd/E26502_01/html/E28388/ennbz.html#scrolltoc
4. ORACLE AMD64 register information: https://docs.oracle.com/cd/E19205-01/821-2506/gituv/index.html
5. http://service.scs.carleton.ca/sivarama/asm_book_web/Student_copies/ch5_addrmodes.pdf


OPCODE is a 200 bit one-hot vector
Operand type is a 3 bit binary vector
Register is a 64 bit one-hot vector


| OPCODE  | Operand type1 |    Base    |    Index   |   Scale   |  Offset   |  Operand type2  |...|
| 200 bit |     3 bit     |   100 bit  |   100 bit  |   1 bit   |  17 bit   |...|
                          |          Register             |                 |...|
                          |           100 bit         | 0 |                 |...|
                          |          string               |                 |...|
                          |           16 bit         | 0  |                 |...|

"""

OPCODE_LEN = 200
OPERAND_TYPE = 3
REGISTER_LEN = 100


# does not support SSE SSE2 and MMX instructions
OPCODE = [
    # Data Movement
    "mov",
    "push",
    "pop",
    "cwtl",
    "cltq",
    "cqto",

    # Unary Operations
    "inc",
    "dec",
    "neg",
    "not",

    # Binary Operations
    "lea",
    "leaq",
    "add",
    "sub",
    "imul",
    "xor",
    "or",
    "and",

    # Shift Operations
    "sal",
    "sar",
    "shr",

    # Special Arithmetic Operations
    "imulq",
    "mulq",
    "idivq",
    "divq",

    # Comparison and Test Instructions
    "cmp",
    "test",

    # Conditional Set Instructions
    "sete",
    "setz",
    "setne",
    "setnz",
    "sets",
    "setns",
    "setg",
    "setnle",
    "setge",
    "setnl",
    "setl",
    "setnge",
    "setle",
    "setng",
    "seta",
    "setnbe",
    "setae",
    "setnb",
    "setbe",
    "setna",

    #Jump Instructions
    "jmp",
    "je",
    "jz",
    "jne",
    "jnz",
    "js",
    "jns",
    "jg",
    "jnle",
    "jge",
    "jnl",
    "jl",
    "jnge",
    "jle",
    "jng",
    "ja",
    "jnbe",
    "jae",
    "jnb",
    "jb",
    "jnae",
    "jbe",
    "jna",

    # Conditional Move Instructions
    "cmove",
    "cmovz",
    "cmovne",
    "cmovenz",
    "cmovs",
    "cmovns",
    "cmovg",
    "cmovnle",
    "cmovge",
    "cmovnl",
    "cmovl",
    "cmovnge",
    "cmovle",
    "cmovng",
    "cmova",
    "cmovnbe",
    "cmovae",
    "cmovnb",
    "cmovb",
    "cmovnae",
    "cmovbe",
    "cmovna",

    # Procedure Call Instruction
    "call",
    "leave",
    "ret",
    "retn"

    # String Instructions
    "cmps",
    "cmpsb",
    "cmpsl",
    "cmpsw",
    "lods",
    "lodsb",
    "lodsl",
    "lodsw",
    "movs",
    "movsb",
    "movsl",
    "movsw",

    # Float point Arithmetic Instructions
    "fabs",
    "fadd",
    "faddp",
    "fchs",
    "fdiv",
    "fdivp",
    "fdivr",
    "fdivrp",
    "fiadd",
    "fidiv",
    "fidivr",
    "fimul",
    "fisub",
    "fisubr",
    "fmul",
    "fmulp",
    "fprem",
    "fprem1",
    "frndint",
    "fscale",
    "fsqrt",
    "fsub",
    "fsubp",
    "fsubr",
    "fsubrp",
    "fxtract",

]

REGISTER = [
    "rax", "eax", "ax", "al", "ah",
    "rcx", "ecx", "cx", "cl", "ch",
    "rdx", "edx", "dx", "dl", "dh",
    "rbx", "ebx", "bx", "bl", "bh", # 20
    "rsi", "esi", "si", "sil",
    "rdi", "edi", "di", "dil",
    "rsp", "esp", "sp", "spl",
    "rbp", "ebp", "bp", "bpl",
    "r8", "r8d", "r8w", "r8b",
    "r9", "r9d", "r9w", "r9b", # 44
    "r10", "r10d", "r10w", "r10b",
    "r11", "r11d", "r11w", "r11b",
    "r12", "r12d", "r12w", "r12b",
    "r13", "r13d", "r13w", "r13b",
    "r14", "r14d", "r14w", "r14b",
    "r15", "r15d", "r15w", "r15b",
    "xmm0", "xmm1", "xmm2", "xmm3", 
    "xmm4", "xmm5", "xmm6", "xmm7", # 76
    "st0", "st1", "st2", "st3",
    "st4", "st5", "st6", "st7", # 84
    "cs", "es", "os", "fs", "gs", "ss", # 90
    "fcw", "fsw", "ftw", "fop", #94
    "frip", "frdp", "mxcsr", "mxcsr_mask",
    "rip", "rflags", # 100
    #normalization
    "string",
    "symbol",
    "address",
    "shl"
]


SEGMENT = [
    "cs",
    "ss",
    "ds",
    "es",
    "fs",
    "gs"
]