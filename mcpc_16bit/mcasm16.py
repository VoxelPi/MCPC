import sys
import numpy as np
import numpy.typing as npt
import pathlib
from mcpc16 import Register, Condition, Operation, encode_instruction, PROGRAM_MEMORY_SIZE
import argparse
from dataclasses import dataclass

ASSEMBLER_SYMBOLS = {
    "true": "1",
    "false": "0",
    "c1": "r14",
    "c2": "r15",
    "r0": "pc",
}

class AssemblySyntaxError(Exception):
    i_source_line: int
    source_line: str
    message: str

    def __init__(self, i_source_line: int, source_line: str, message: str):
        super().__init__(f"Failed to parse line {i_source_line + 1} '{source_line}'. {message}")
        self.i_source_line = i_source_line
        self.source_line = source_line
        self.message = message

def parse_register(text: str) -> Register | None:  
    text = text.upper()
    try:
        register = Register[text]
        return register
    except KeyError:
        return None
    
def is_register(text: str) -> bool:
    text = text.upper()
    try:
        _ = Register[text]
        return True
    except KeyError:
        return False
    
def parse_value(text: str) -> Register | np.uint16 | None:
    # Check if value is a register.
    register = parse_register(text)
    if register is not None:
        return register
    
    # Check if value is an immediate value.
    immediate_value = parse_immediate_value(text)
    if immediate_value is not None:
        return immediate_value
    
    # Invalid value
    return None

def is_value(text: str) -> bool:
    # Check if value is a register.
    register = parse_register(text)
    if register is not None:
        return True
    
    # Check if value is an immediate value.
    immediate_value = parse_immediate_value(text)
    if immediate_value is not None:
        return True
    
    # Invalid value
    return False

def parse_immediate_value(text: str) -> np.uint16 | None:
    try:
        return np.uint16(int(text, 0) & 0xFFFF)
    except ValueError:
        return None

def is_immediate_value(text: str) -> bool:
    try:
        _ = int(text, 0)
        return True
    except ValueError:
        return False
    
condition_operators = {
    "=": Condition.EQUAL,
    "!=": Condition.NOT_EQUAL,
    ">": Condition.GREATER,
    "<": Condition.LESS,
    ">=": Condition.GREATER_OR_EQUAL,
    "<=": Condition.LESS_OR_EQUAL,
}

# Binary operators.
# These are operations with the following syntax: <output> = <a> <operator> <b>
binary_operations = {
    'and': Operation.AND,
    'nand': Operation.NAND,
    'or': Operation.OR,
    'nor': Operation.NOR,
    'xor': Operation.XOR,
    'xnor': Operation.XNOR,
    '+': Operation.ADD,
    '-': Operation.SUB,
    '*': Operation.MULTIPLY,
    '/': Operation.DIVIDE,
    '%': Operation.MODULO,
}

# Unary operators.
# These are operations with the following syntax: <output> = <operator> <input>
# Internally the input is send to A as well as B.
unary_operators = {
    'not': Operation.NAND,
    'inc': Operation.INC,
    'dec': Operation.DEC,
    'sqrt': Operation.SQRT,
}

# Instructions that take no arguments. Internally the register R1 is used as A, B and output
no_args_instructions = {
    'nop': Operation.AND,
    'break': Operation.BREAK,
}

@dataclass
class AssembledProgram():
    binary: npt.NDArray[np.uint64]
    text: list[str]
    instructions: list[str]
    src_mapping: list[int] # Instruction id to code line.
    labels: dict[str, int]

    @property
    def n_instructions(self) -> int:
        return len(self.instructions)

def _parse_instruction(instruction_text: str, i_instruction: int, source_line: str, i_source_line: int) -> np.uint64:
    instruction_parts = instruction_text.split(" ")
    n_instruction_parts = len(instruction_parts)

    # Handle conditions
    if "if" in instruction_parts:
        if_index = instruction_parts.index("if")
        condition_parts = instruction_parts[(if_index + 1):]
        n_condition_parts = len(condition_parts)
        instruction_parts = instruction_parts[:if_index]
        n_instruction_parts = len(instruction_parts)

        # Check general condition syntax.
        if n_condition_parts != 3 or condition_parts[2] != "0":
            raise AssemblySyntaxError(i_source_line, source_line, "Invalid condition syntax. Should be 'if <register> <condition> 0'.")
        
        # Get condition.
        if condition_parts[1] not in condition_operators:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition operator '{condition_parts[1]}'.")
        condition = condition_operators[condition_parts[1]]

        # Get condition source register.
        condition_register = parse_register(condition_parts[0])
        if condition_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition source register '{condition_parts[0]}'.")

    else:
        condition_register = Register.R14
        condition = Condition.ALWAYS

    # RETURN
    if instruction_text == "return":
        return encode_instruction(Operation.STACK_POP, condition_register, condition, Register.PC, 0, 0)

    # NO ARGS
    if instruction_text in no_args_instructions:
        operation = no_args_instructions[instruction_text]
        return encode_instruction(operation, condition_register, Condition.NEVER, Register.R1, Register.R1, Register.R1)

    # JUMP
    if instruction_parts[0] == 'jump':
        # Check general syntax.
        if n_instruction_parts != 2:
            raise AssemblySyntaxError(i_source_line, source_line, "Invalid syntax for jump. Should be 'jump <register|immediate>'.")

        # Parse the jump target value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid jump target '{instruction_parts[1]}'.")
        
        # Encode instruction.
        return encode_instruction(Operation.A, condition_register, condition, Register.PC, value, 0)
        
    # SKIP
    if instruction_parts[0] == 'skip':
        # Check general syntax.
        if n_instruction_parts != 2:
            raise AssemblySyntaxError(i_source_line, source_line, "Invalid syntax for skip. Should be 'skip <register|immediate>'.")

        # Parse the skip value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid skip length '{instruction_parts[1]}'.")
        
        # Encode instruction.
        return encode_instruction(Operation.ADD, condition_register, condition, Register.PC, value, Register.PC)

    # Assignments, that are instructions of the form '<output_register> = ...'
    if n_instruction_parts >= 2 and is_register(instruction_parts[0]) and instruction_parts[1] == "=":
        # Get the output register
        output_register = parse_register(instruction_parts[0])
        if output_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register '{instruction_parts[0]}'.")

        # LOAD
        if n_instruction_parts == 3:
            value_text = instruction_parts[2]

            # value, '<r> = <value>'
            value = parse_value(value_text)
            if value is not None:
                return encode_instruction(Operation.A, condition_register, condition, output_register, value, Register.R1)

            # memory, '<r> = [<address>]'
            if value_text[0] == "[" and value_text[-1] == "]":
                # Parse the address
                address_text = value_text[1:-1]
                address = parse_value(address_text)
                if address is None:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value for memory address: '{address_text}'.")

                # Encode instruction.
                return encode_instruction(Operation.MEMORY_LOAD, condition_register, condition, output_register, address, 0)

            # io poll, '<r> = poll'
            if value_text == "poll":
                return encode_instruction(Operation.IO_POLL, condition_register, condition, output_register, 0, 0)

            # io read, '<r> = read'
            if value_text == "read":
                return encode_instruction(Operation.IO_READ, condition_register, condition, output_register, 0, 0)
            
            # stack peek, '<r> = peek'
            if value_text == "peek":
                return encode_instruction(Operation.STACK_PEEK, condition_register, condition, output_register, 0, 0)
            
            # stack pop, '<r> = pop'
            if value_text == "pop":
                return encode_instruction(Operation.STACK_POP, condition_register, condition, output_register, 0, 0)

            # Unsupported load value.
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid load value: '{value_text}'.")

        # SHIFT
        # TODO: shift count?
        if n_instruction_parts == 5 and instruction_parts[2] == "shift":
            # Parse value.
            value = parse_value(instruction_parts[4])
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value '{instruction_parts[4]}'.")

            # Parse direction.
            match instruction_parts[3]:
                case "left":
                    return encode_instruction(Operation.SHIFT_LEFT, condition_register, condition, output_register, value, 1)
                case "right":
                    return encode_instruction(Operation.SHIFT_RIGHT, condition_register, condition, output_register, value, 1)
                case _:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid shift direction '{instruction_parts[3]}'.")
        
        # ROTATE
        if n_instruction_parts == 5 and instruction_parts[2] == "rotate":
            # Parse value.
            value = parse_value(instruction_parts[4])
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value '{instruction_parts[4]}'.")

            # Parse direction.
            match instruction_parts[3]:
                case "left":
                    return encode_instruction(Operation.ROTATE_LEFT, condition_register, condition, output_register, value, 1)
                case "right":
                    return encode_instruction(Operation.ROTATE_RIGHT, condition_register, condition, output_register, value, 1)
                case _:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid rotation direction '{instruction_parts[3]}'.")

        # # CHECK
        # if n_instruction_parts == 6 and instruction_parts[2] == "check":
        #     # Parse value.
        #     value = parse_value(instruction_parts[3])
        #     if value is None:
        #         raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value '{instruction_parts[4]}'.")

        #     # Parse condition operator.
        #     if instruction_parts[4] not in condition_operators:
        #         raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition '{instruction_parts[4]}' for check.")
        #     condition = condition_operators[instruction_parts[4]]
        #     operation = operation_check(condition)

        #     # Check 0 argument.
        #     if instruction_parts[5] != "0":
        #         raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition '{instruction_parts[4]} {instruction_parts[5]}' for check.")
            
        #     # Encode instruction.
        #     return encode_instruction(operation, condition_register, condition, output_register, value, 0)
        
        # Bit operation, '<r> = <value> bit <operation> <bit>'
        if n_instruction_parts == 6 and instruction_parts[3] == "bit":
            # Parse value.
            value = parse_value(instruction_parts[2])
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value '{instruction_parts[2]}'.")
            
            # Parse bit.
            bit = parse_value(instruction_parts[5])
            if bit is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid bit '{instruction_parts[5]}'.")
            
            # Parse bit operation.
            bit_operation = instruction_parts[4]
            match bit_operation:
                case "get":
                    return encode_instruction(Operation.BIT_GET, condition_register, condition, output_register, value, bit)
                case "set":
                    return encode_instruction(Operation.BIT_SET, condition_register, condition, output_register, value, bit)
                case "clear":
                    return encode_instruction(Operation.BIT_CLEAR, condition_register, condition, output_register, value, bit)
                case "toggle":
                    return encode_instruction(Operation.BIT_TOGGLE, condition_register, condition, output_register, value, bit)
                case _:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid bit operation '{bit_operation}'.")

        # Unary operators
        if n_instruction_parts == 4 and instruction_parts[2] in unary_operators:
            # Parse value.
            value = parse_value(instruction_parts[3])
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value '{instruction_parts[3]}'.")
            
            # Parse operation.
            operation = unary_operators[instruction_parts[2]]

            # Encode instruction.
            return encode_instruction(operation, condition_register, condition, output_register, value, 0)

        # Binary operators
        if n_instruction_parts == 5 and instruction_parts[3] in binary_operations:
            # Parse value for A.
            value_a = parse_value(instruction_parts[2])
            if value_a is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value for A: '{instruction_parts[3]}'.")
            
            # Parse value for B.
            value_b = parse_value(instruction_parts[4])
            if value_b is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value for A: '{instruction_parts[3]}'.")

            # Parse operation.
            operation = binary_operations[instruction_parts[3]]

            # Encode instruction.
            return encode_instruction(operation, condition_register, condition, output_register, value_a, value_b)

        raise AssemblySyntaxError(i_source_line, source_line, "Invalid right hand side for assignment")

    # Memory store, '[<address>] = <value>'
    if n_instruction_parts == 3 and instruction_parts[0][0] == "[" and instruction_parts[0][-1] == "]" and instruction_parts[1] == "=":
        # Parse address.
        address_text = instruction_parts[0][1:-1]
        address = parse_value(address_text)
        if address is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid memory address: '{address_text}'.")
        
        # Parse value.
        value = parse_value(instruction_parts[2])
        if value is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value: '{instruction_parts[2]}'.")
    
        # Encode instruction.
        return encode_instruction(Operation.MEMORY_STORE, condition_register, Condition.NEVER, Register.R1, address, value)

    # Stack push, 'push <value>'
    if n_instruction_parts == 2 and instruction_parts[0] == "push":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value: '{value}'.")

        # Encode instruction.
        return encode_instruction(Operation.STACK_PUSH, condition_register, Condition.NEVER, Register.R1, value, 0)

    # Call, 'call <address>'
    if n_instruction_parts == 2 and instruction_parts[0] == "call":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value: '{value}'.")

        # Encode instruction.
        return encode_instruction(Operation.STACK_CALL, condition_register, condition, Register.PC, Register.PC, value)

    # IO WRITE, 'write <value>'
    if n_instruction_parts == 2 and instruction_parts[0] == "write":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value: '{value}'.")

        # Encode instruction.
        return encode_instruction(Operation.IO_WRITE, condition_register, Condition.NEVER, Register.R1, value, 0)
    
    # No instruction pattern detected, throw generic syntax exception.
    raise AssemblySyntaxError(i_source_line, source_line, "Invalid instruction")

def assemble(src_lines: list[str], default_macro_symbols: dict[str, str] = {}) -> AssembledProgram | None:
    # Remove comments
    code_lines = np.array([line.split('#', 1)[0].strip() for line in src_lines])

    if len(code_lines) == 0:
        return AssembledProgram(np.zeros(PROGRAM_MEMORY_SIZE, dtype=np.uint64), src_lines, code_lines.tolist(), [], {})

    # Remove empty lines
    src_to_code_line = np.roll(np.cumsum(code_lines != ""), 1)
    src_to_code_line[0] = 0
    code_line_mapping: list[int] = np.zeros(np.max(src_to_code_line) + 1).tolist()
    for i, line in enumerate(src_to_code_line):
        code_line_mapping[line] = i
    code_lines = code_lines[code_line_mapping]
    if code_lines[-1] == '':
        code_lines = code_lines[:-1]

    # Remove multiple whitespace
    code_lines = [" ".join(line.split()).lower() for line in code_lines]

    instruction_lines: list[str] = []
    instruction_mapping: list[int] = []
    labels: dict[str, int] = {}
    macro_symbols: dict[str, str] = default_macro_symbols

    for i_code_line, line in enumerate(code_lines):
        i_src_line = code_line_mapping[i_code_line]
        if line.startswith("@"):
            label_id = line[1:]
            if label_id in labels:
                raise Exception(f"Trying to declare used label '{label_id}' in line {i_src_line + 1}.\nPreviously declared in line {instruction_mapping[labels[label_id]]+1-1}")

            labels[label_id] = len(instruction_lines)
            continue

        if line.startswith("!define"):
            line_parts = line.split(" ", 2)
            if len(line_parts) < 3:
                raise Exception(f"Invalid define in line {i_src_line+1}")
            symbol = line_parts[1]
            value = line_parts[2]
            macro_symbols[symbol] = value
            continue

        instruction_lines.append(line)
        instruction_mapping.append(i_src_line)
    n_instructions = len(instruction_lines)

    for label, instruction_id in labels.items():
        if instruction_id >= n_instructions:
            raise Exception(f"The label {label} marks no instruction.")

    if len(instruction_lines) > 0:
        # Replace used labels
        instructions_text = "\n".join([f" {line} " for line in instruction_lines])
        for label, instruction_id in labels.items():
            instructions_text = instructions_text.replace(f" @{label} ", f" {instruction_id} ")
            instructions_text = instructions_text.replace(f"[@{label}]", f"[{instruction_id}]")

        # Replace symbols
        symbols = ASSEMBLER_SYMBOLS | macro_symbols
        for symbol, value in symbols.items():
            instructions_text = instructions_text.replace(f" {symbol} ", f" {value} ")
            instructions_text = instructions_text.replace(f"[{symbol}]", f"[{value}]")
        instruction_lines = instructions_text.split("\n")
        instruction_lines = [ line.strip() for line in instruction_lines ]

    instructions = np.zeros(n_instructions, dtype=np.uint64)
    for i_instruction, instruction_text in enumerate(instruction_lines):
        i_src_line = instruction_mapping[i_instruction]
        src_line = src_lines[i_src_line]
        instruction_text = instruction_text.lower()
        try:
            # Parse the instruction
            instructions[i_instruction] = _parse_instruction(instruction_text, i_instruction, src_line, i_src_line)
        except AssemblySyntaxError as syntax_error:
            print(syntax_error, file=sys.stderr)
            return None

        except Exception as exception:
            print(f"Exception whilst parsing line {i_src_line + 1}: '{src_line}'.")
            raise exception

    return AssembledProgram(instructions, src_lines, instruction_lines, instruction_mapping, labels)

def program_to_memory(instructions: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    n_instructions = len(instructions)
    memory = np.zeros(PROGRAM_MEMORY_SIZE, dtype=np.uint64)
    memory[0:n_instructions] = instructions[0:n_instructions]
    return memory

# MAIN PROGRAM
if __name__ == "__main__":
    # Parse arguments
    argument_parser = argparse.ArgumentParser(
        prog="MCASM",
        description="Assembler for the MCPC",
    )
    
    argument_parser.add_argument("filename", nargs="?", default="./mcpc_16bit/programs/program.mcasm")
    argument_parser.add_argument("-o", "--output")
    argument_parser.add_argument("-c", "--check", action="store_true")
    arguments = argument_parser.parse_args()
    input_filename: str = arguments.filename
    output_filename: str | None = arguments.output
    check_mode: bool = arguments.check or False

    # Resolve input and output file path.
    input_filepath = pathlib.Path(input_filename)
    if output_filename is None:
        output_filename = f"{input_filepath.stem}.mcbin"
    output_filepath = pathlib.Path(output_filename)

    # Read input lines
    with open(input_filepath, "r") as input_file:
        src_lines = [line.strip() for line in input_file.readlines()]

    # Assemble the program
    program = assemble(src_lines)
    if program is None:
        print("Failed to assemble program.", file=sys.stderr)
        exit(1)

    # Write the output to a file.
    if not check_mode:
        with open(output_filepath, "wb") as output_file:
            output_file.write(program.binary.tobytes())

    print(f"Assembled {len(program.instructions)} instructions")
