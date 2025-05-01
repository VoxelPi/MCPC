from enum import Enum
import numpy as np

PROGRAM_MEMORY_SIZE = 256

class InstructionType(Enum):
    LOAD = 0b0000_0000_0000_0000
    EXEC = 0b1000_0000_0000_0000

class Register(Enum):
    PC = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7

class Operation(Enum):
    CLEAR = 0
    FILL = 1
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

    MEMORY_LOAD = 25
    MEMORY_STORE = 26

    IO_POLL = 27
    IO_READ = 28
    IO_WRITE = 29

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

def opcode_load_instruction(relative: bool, condition: Condition, output_reg: Register, value: int) -> np.uint16:
    return np.uint16(InstructionType.LOAD.value | (output_reg.value << 12) | (condition.value << 9) | ((1 if relative else 0) << 8) | (value & 0xFF))

def opcode_exec_instruction(input_a_reg: Register, input_b_reg: Register, output_reg: Register, operation: Operation) -> np.uint16:
    return np.uint16(InstructionType.EXEC.value | (output_reg.value << 12) | (input_b_reg.value << 9) | (input_a_reg.value << 6) | operation.value)