from enum import Enum
import numpy as np

PROGRAM_MEMORY_SIZE = 0xFFFF

class Register(Enum):
    PC = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11
    R12 = 12
    R13 = 13
    R14 = 14
    R15 = 15

class Operation(Enum):
    A = 0
    NOT_A = 1
    AND = 2
    NAND = 3
    OR = 4
    NOR = 5
    XOR = 6
    XNOR = 7

    INC = 8
    DEC = 9
    ADD = 10
    SUB = 11

    SHIFT_LEFT = 12
    SHIFT_RIGHT = 13
    ROTATE_LEFT = 14
    ROTATE_RIGHT = 15

    CHECK_ALWAYS = 16
    CHECK_NEVER = 17
    CHECK_EQUAL = 18
    CHECK_NOT_EQUAL = 19
    CHECK_LESS = 20
    CHECK_GREATER_OR_EQUAL = 21
    CHECK_GREATER = 22
    CHECK_LESS_OR_EQUAL = 23

    MEMORY_LOAD = 24
    MEMORY_STORE = 25

    IO_POLL = 26
    IO_READ = 27
    IO_WRITE = 28

    STACK_PUSH = 29
    STACK_PEEK = 30
    STACK_POP = 31

    BIT_GET = 32
    BIT_SET = 33
    BIT_CLEAR = 34
    BIT_TOGGEL = 35

    UNDEFINED_36 = 36
    UNDEFINED_37 = 37
    UNDEFINED_38 = 38
    UNDEFINED_39 = 39
    UNDEFINED_40 = 40
    UNDEFINED_41 = 41
    UNDEFINED_42 = 42
    UNDEFINED_43 = 43
    UNDEFINED_44 = 44
    UNDEFINED_45 = 45
    UNDEFINED_46 = 46
    UNDEFINED_47 = 47

    MULTIPLY = 48
    DIVIDE = 49
    MODULO = 50
    SQRT = 51
    UNDEFINED_52 = 52
    UNDEFINED_53 = 53
    UNDEFINED_54 = 54
    UNDEFINED_55 = 55
    UNDEFINED_56 = 56
    UNDEFINED_57 = 57
    UNDEFINED_58 = 58
    UNDEFINED_59 = 59
    UNDEFINED_60 = 60
    UNDEFINED_61 = 61
    UNDEFINED_62 = 62
    BREAK = 63

class Condition(Enum):
    ALWAYS = 0
    NEVER = 1
    EQUAL = 2
    NOT_EQUAL = 3
    LESS = 4
    GREATER_OR_EQUAL = 5
    GREATER = 6
    LESS_OR_EQUAL = 7

def operation_check(condition: Condition) -> Operation:
    return list(Operation)[Operation.CHECK_ALWAYS.value + condition.value]

def is_valid_condition_source(register: Register) -> bool:
    match register:
        case Register.R14 | Register.R15:
            return True
        case _:
            return False

def condition_source(register: Register) -> int:
    match register:
        case Register.R14:
            return 0
        case Register.R15:
            return 1
        case _:
            raise Exception(f"Register {register} can't be used for conditions")

def generate_opcode(
    operation: Operation, 
    condition_register: Register, 
    condition: Condition, 
    output: Register, 
    a: Register | np.uint16 | int, 
    b: Register | np.uint16 | int,
) -> np.uint32:
    if isinstance(a, Register):
        a_value = np.uint32(a.value)
        a_mode = np.uint32(1)
    else:
        a_value = np.uint32(a)
        a_mode = np.uint32(0)

    if isinstance(b, Register):
        b_value = np.uint32(b.value)
        b_mode = np.uint32(1)
    else:
        b_value = np.uint32(b)
        b_mode = np.uint32(0)
    
    opcode = np.uint32(0)
    opcode |= np.uint32(a_mode) << 0 # A source
    opcode |= np.uint32(b_mode) << 1 # B source
    opcode |= np.uint32(output.value) << 2 # Output address
    opcode |= np.uint32(condition.value) << 6 # Condition
    opcode |= np.uint32(condition_source(condition_register) & 0b1) << 9 # Condition source
    opcode |= np.uint32(operation.value) << 10 # Operation
    opcode |= np.uint32(a_value) << 16 # A value
    opcode |= np.uint32(b_value) << 32 # B Value
    return np.uint32(opcode)