from enum import Enum
import numpy as np
import pathlib
from mcpc import *
import argparse
from dataclasses import dataclass

ASSEMBLER_SYMBOLS = {
    "true": "1",
    "false": "0",
    "rc": "r7"
}

def parse_register(text: str) -> bool:
    if len(text) != 2:
        return None
    
    text = text.upper()
    if text == "PC":
        return Register.PC
    if text == "RC":
        return Register.R7
    if text[0] == "R" and text[1].isdigit() and (int(text[1]) in range(8)):
        return list(Register)[int(text[1])]
    return None
    
def is_register_id(text: str) -> Register | None:
    if len(text) != 2:
        return False
    
    text = text.upper()
    if text == "PC":
        return True
    if text == "RC":
        return True
    if text[0] == "R" and text[1].isdigit() and (int(text[1]) in range(8)):
        return True
    return False

def int_or_none(text: str) -> int | None:
    try:
        return int(text, 0)
    except ValueError:
        return None
    
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
    binary: np.ndarray[np.uint16]
    text: list[str]
    instructions: list[str]
    src_mapping: dict[int, int] # Instruction id to code line.
    labels: dict[str, int]

    @property
    def n_instructions(self) -> int:
        return len(self.instructions)

def assemble(src_lines: list[str], default_macro_symbols: dict[str, str] = {}) -> AssembledProgram:
    # Remove comments
    code_lines = np.array([line.split('#', 1)[0].strip() for line in src_lines])

    if len(code_lines) == 0:
        return AssembledProgram(np.zeros(256, dtype=np.uint16), src_lines, code_lines, dict(), dict())

    # Remove empty lines
    src_to_code_line = np.roll(np.cumsum(code_lines != ""), 1)
    src_to_code_line[0] = 0
    code_line_mapping: list[int] = np.zeros(np.max(src_to_code_line) + 1).tolist()
    for i, line in enumerate(src_to_code_line):
        code_line_mapping[line] = i
    code_lines = code_lines[code_line_mapping]

    # Remove multiple whitespace
    code_lines = [" ".join(line.split()) for line in code_lines]

    instruction_lines: list[str] = []
    instruction_mapping: list[int] = []
    labels: dict[str, int] = {}
    macro_symbols: dict[str, str] = default_macro_symbols

    for i_code_line, line in enumerate(code_lines):
        i_src_line = code_line_mapping[i_code_line]
        if line.startswith("@"):
            label_id = line[1:]
            if label_id in labels:
                raise Exception(f"Trying to declare used label '{label_id}' in line {i_src_line+1}.\nPreviously declared in line {instruction_mapping[labels[label_id]]+1-1}")

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
            raise Exception(f"The label {label} marks no instruction")

    if len(instruction_lines) > 0:
        # Replace used labels
        instructions_text = "\n".join(instruction_lines)
        for label, instruction_id in labels.items():
            instructions_text = instructions_text.replace(f"@{label}", f"{instruction_id}")

        # Replace symbols
        symbols = ASSEMBLER_SYMBOLS | macro_symbols
        for symbol, value in symbols.items():
            instructions_text = instructions_text.replace(symbol, value)
        instruction_lines = instructions_text.split("\n")

    instructions = np.zeros(n_instructions, dtype=np.uint16)
    for i_instruction, instruction_text in enumerate(instruction_lines):
        i_src_line = instruction_mapping[i_instruction]
        src_line = src_lines[i_src_line]
        instruction_text = instruction_text.lower()
        instruction_parts = instruction_text.split(" ")
        n_instruction_parts = len(instruction_parts)
        try:

            # NO ARGS
            if instruction_text in no_args_instructions:
                operation = no_args_instructions[instruction_text]
                instructions[i_instruction] = opcode_exec_instruction(Register.R1, Register.R1, Register.R1, operation)
                continue

            # JUMP
            if instruction_parts[0] == 'jump':

                # UNCONDITONAL JUMP
                if n_instruction_parts == 2:
                    # To register value.
                    if is_register_id(instruction_parts[1]):
                        input_register = parse_register(instruction_parts[1])
                        instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, Register.PC, Operation.AND)
                        continue

                    # To immediate value.
                    value = int_or_none(instruction_parts[1])
                    if value is None:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for jump target: '{instruction_parts[1]}'")
                    instructions[i_instruction] = opcode_load_instruction(False, Condition.ALWAYS, Register.PC, value)
                    continue

                # CONDITIONAL JUMP
                if n_instruction_parts == 6 and instruction_parts[2] == "if" and instruction_parts[3] in ("r7", "rc") and instruction_parts[4] in condition_operators and instruction_parts[5] == '0':
                    value = int_or_none(instruction_parts[1])
                    if value is None:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for jump target: '{instruction_parts[1]}'")
                    
                    condition = condition_operators[instruction_parts[4]]
                    instructions[i_instruction] = opcode_load_instruction(False, condition, Register.PC, value)
                    continue

                raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for jump: '{src_line}'")
            
            # SKIP
            if instruction_parts[0] == 'skip':

                # UNCONDITONAL SKIP
                if n_instruction_parts == 2:
                    # By register value.
                    if is_register_id(instruction_parts[1]):
                        input_register = parse_register(instruction_parts[1])
                        instructions[i_instruction] = opcode_exec_instruction(Register.PC, input_register, Register.PC, Operation.ADD)
                        continue

                    # By immediate value.
                    value = int_or_none(instruction_parts[1])
                    if value is None:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for skip length: '{instruction_parts[1]}'")
                    instructions[i_instruction] = opcode_load_instruction(True, Condition.ALWAYS, Register.PC, value)
                    continue

                # CONDITIONAL SKIP
                if n_instruction_parts == 6 and instruction_parts[2] == "if" and instruction_parts[3] == "r7" and instruction_parts[4] in condition_operators and instruction_parts[5] == '0':
                    value = int_or_none(instruction_parts[1])
                    if value is None:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for skip length: '{instruction_parts[1]}'")
                    
                    condition = condition_operators[instruction_parts[4]]
                    instructions[i_instruction] = opcode_load_instruction(True, condition, Register.PC, value)
                    continue

                raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for skip: '{src_line}'")

            # RELATIVE LOADS
            if n_instruction_parts >= 2 and is_register_id(instruction_parts[0]) and instruction_parts[1] == "+=":
                output_register = parse_register(instruction_parts[0])

                # RELATIVE LOAD
                if n_instruction_parts == 3:
                    # By register value.
                    if is_register_id(instruction_parts[2]):
                        input_register = parse_register(instruction_parts[2])
                        instructions[i_instruction] = opcode_exec_instruction(output_register, input_register, output_register, Operation.ADD)
                        continue

                    # By immediate value.
                    value = int_or_none(instruction_parts[2])
                    if value is None:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for load value: '{instruction_parts[2]}'") 
                    instructions[i_instruction] = opcode_load_instruction(True, Condition.ALWAYS, output_register, value)
                    continue

                # RELATIVE CONDITIONAL LOAD
                if n_instruction_parts == 7 and instruction_parts[3] == "if" and instruction_parts[4] == "r7" and instruction_parts[5] in condition_operators and instruction_parts[6] == '0':
                    value = int_or_none(instruction_parts[2])
                    if value is None:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for load value: '{instruction_parts[2]}'")
                
                    condition = condition_operators[instruction_parts[5]]
                    instructions[i_instruction] = opcode_load_instruction(True, condition, output_register, value)
                    continue

            # Assignments
            if n_instruction_parts >= 2 and is_register_id(instruction_parts[0]) and instruction_parts[1] == "=":
                output_register = parse_register(instruction_parts[0])
                # LOAD
                if n_instruction_parts == 3:
                    # RANDOM VALUE
                    if instruction_parts[2] == "random":
                        instructions[i_instruction] = opcode_exec_instruction(Register.R1, Register.R1, output_register, Operation.RANDOM)
                        continue

                    # FROM ANOTHER REGISTER
                    if is_register_id(instruction_parts[2]):
                        input_register = parse_register(instruction_parts[2])
                        instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, Operation.AND)
                        continue

                    # INTERMEDIATE VALUE
                    value = int_or_none(instruction_parts[2])
                    if value is not None:
                        instructions[i_instruction] = opcode_load_instruction(False, Condition.ALWAYS, output_register, value)
                        continue

                    # FROM MEMORY
                    if instruction_parts[2][0] == "[" and instruction_parts[2][-1] == "]":
                        address_register_id = instruction_parts[2][1:-1]
                        if not is_register_id(address_register_id):
                            raise Exception(f"Failed to parse line {i_src_line}: Invalid register for memory address: '{address_register_id}'")
                        address_register = parse_register(address_register_id)

                        instructions[i_instruction] = opcode_exec_instruction(address_register, address_register, output_register, Operation.MEMORY_LOAD)
                        continue

                    # FROM IO POLL
                    if n_instruction_parts == 3 and instruction_parts[2] == "poll":
                        instructions[i_instruction] = opcode_exec_instruction(output_register, output_register, output_register, Operation.IO_POLL)
                        continue

                    # FROM IO READ
                    if n_instruction_parts == 3 and instruction_parts[2] == "read":
                        instructions[i_instruction] = opcode_exec_instruction(output_register, output_register, output_register, Operation.IO_READ)
                        continue

                    raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for load value: '{instruction_parts[2]}'")

                # CONDITIONAL LOAD
                if n_instruction_parts == 7 and instruction_parts[3] == "if" and instruction_parts[4] in ("r7", "rc") and instruction_parts[5] in condition_operators and instruction_parts[6] == '0':
                    value = int_or_none(instruction_parts[2])
                    if value is None:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for load value: '{instruction_parts[2]}'")
                
                    condition = condition_operators[instruction_parts[5]]
                    instructions[i_instruction] = opcode_load_instruction(False, condition, output_register, value)
                    continue

                # BIT GET
                if n_instruction_parts == 5 and is_register_id(instruction_parts[2]) and instruction_parts[3] == "bit":
                    input_register = parse_register(instruction_parts[2])
                    value = int_or_none(instruction_parts[4])
                    if value < 0 or value > 7:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid bit index {value}")
                    
                    instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, list(Operation)[Operation.BIT_GET_0.value + value])
                    continue

                # SHIFT
                if n_instruction_parts == 5 and instruction_parts[2] == "shift" and is_register_id(instruction_parts[4]):
                    input_register = parse_register(instruction_parts[4])
                    if instruction_parts[3] == "left":
                        instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, Operation.SHIFT_LEFT)
                        continue

                    if instruction_parts[3] == "right":
                        instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, Operation.SHIFT_RIGHT)
                        continue

                    raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for shift: '{src_line}'") 
                
                # ROTATE
                if n_instruction_parts == 5 and instruction_parts[2] == "rotate" and is_register_id(instruction_parts[4]):
                    input_register = parse_register(instruction_parts[4])
                    if instruction_parts[3] == "left":
                        instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, Operation.ROTATE_LEFT)
                        continue

                    if instruction_parts[3] == "right":
                        instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, Operation.ROTATE_RIGHT)
                        continue

                    raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for shift: '{src_line}'") 

                # CHECK
                if n_instruction_parts == 6 and instruction_parts[2] == "check":
                    if not is_register_id(instruction_parts[3]):
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for check: '{src_line}'") 
                    input_register = parse_register(instruction_parts[3])

                    if instruction_parts[4] not in condition_operators:
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid condition '{instruction_parts[4]}' for check: '{src_line}'")
                    condition = condition_operators[instruction_parts[4]]

                    if instruction_parts[5] != "0":
                        raise Exception(f"Failed to parse line {i_src_line}: Invalid condition '{instruction_parts[4]} {instruction_parts[5]}' for check: '{src_line}'")
                    
                    instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, operation_check(condition))
                    continue

                # MAC
                if n_instruction_parts == 5 and instruction_parts[2] == "mac" and is_register_id(instruction_parts[3]) and is_register_id(instruction_parts[4]):
                    input_a_register = parse_register(instruction_parts[3])
                    input_b_register = parse_register(instruction_parts[4])
                    
                    instructions[i_instruction] = opcode_exec_instruction(input_a_register, input_b_register, output_register, Operation.MULTIPLY_ACCUMULATE)
                    continue

                # MAC RS
                if n_instruction_parts == 5 and instruction_parts[2] == "macrs" and instruction_parts[3] == "adc" and is_register_id(instruction_parts[4]):
                    input_register = parse_register(instruction_parts[4])
                    
                    instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, Operation.MULTIPLY_ACCUMULATE_RESET)
                    continue

                # UNARY OPERATIONS
                if n_instruction_parts == 4 and instruction_parts[2] in unary_operators and is_register_id(instruction_parts[3]):
                    input_register = parse_register(instruction_parts[3])
                    operation = unary_operators[instruction_parts[2]]

                    instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, output_register, operation)
                    continue

                # BINARY OPERATIONS
                if n_instruction_parts == 5 and instruction_parts[3] in binary_operations and is_register_id(instruction_parts[2]) and is_register_id(instruction_parts[4]):
                    input_a_register = parse_register(instruction_parts[2])
                    input_b_register = parse_register(instruction_parts[4])
                    operation = binary_operations[instruction_parts[3]]

                    instructions[i_instruction] = opcode_exec_instruction(input_a_register, input_b_register, output_register, operation)
                    continue

                raise Exception(f"Failed to parse line {i_src_line}: Invalid syntax for assignment: '{src_line}'")

            # BIT SET
            if n_instruction_parts == 5 and is_register_id(instruction_parts[0]) and instruction_parts[1] == "bit" and instruction_parts[3] == "=" and is_register_id(instruction_parts[4]):
                output_register = parse_register(instruction_parts[0])
                input_register = parse_register(instruction_parts[4])
                value = int_or_none(instruction_parts[2])
                if value < 0 or value > 7:
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid bit index {value}")
                
                instructions[i_instruction] = opcode_exec_instruction(input_register, output_register, output_register, list(Operation)[Operation.BIT_SET_0.value + value])
                continue

            # STORE
            if n_instruction_parts == 3 and instruction_parts[0][0] == "[" and instruction_parts[0][-1] == "]" and instruction_parts[1] == "=":
                address_register_id = instruction_parts[0][1:-1]
                if not is_register_id(address_register_id):
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid register for memory address: '{address_register_id}'")
                address_register = parse_register(address_register_id)

                if not is_register_id(instruction_parts[2]):
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid register for memory value: '{instruction_parts[2]}'")
                input_register = parse_register(instruction_parts[2])

                instructions[i_instruction] = opcode_exec_instruction(address_register, input_register, input_register, Operation.MEMORY_STORE)
                continue

            # CLEAR
            if instruction_parts[0] == "clear":
                if len(instruction_parts) != 2:
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid arguments, syntax: 'clear <register>C'")
                
                output_register = parse_register(instruction_parts[1])
                if output_register == None:
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid register '{instruction_parts[1]}'")
                
                instructions[i_instruction] = opcode_exec_instruction(Register.R1, Register.R1, output_register, Operation.CLEAR)
                continue

            # FILL
            if instruction_parts[0] == "fill":
                if len(instruction_parts) != 2:
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid arguments, syntax: 'fill <register>'")
                
                output_register = parse_register(instruction_parts[1])
                if output_register == None:
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid register '{instruction_parts[1]}'")
                
                instructions[i_instruction] = opcode_exec_instruction(Register.R1, Register.R1, output_register, Operation.FILL)
                continue

            # CONFIGURE
            if instruction_parts[0] == "configure":
                if len(instruction_parts) < 2:
                    raise Exception(f"Failed to parse line {i_src_line}: Missing configuration target.")
                
                if instruction_parts[1] == "dac":
                    if len(instruction_parts) < 3 or not is_register_id(instruction_parts[2]):
                        raise Exception(f"Failed to parse line {i_src_line}: DAC configuration requires one input.")
                    
                    input_register = parse_register(instruction_parts[2])
                    instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, input_register, Operation.CONFIGURE_DAC)
                    continue
                
                raise Exception(f"Failed to parse line {i_src_line}: Invalid configuration target '{instruction_parts[1]}'")

            # IO WRITE
            if n_instruction_parts == 2 and instruction_parts[0] == "write":
                if not is_register_id(instruction_parts[1]):
                    raise Exception(f"Failed to parse line {i_src_line}: Invalid input register '{instruction_parts[1]}'")
                input_register = parse_register(instruction_parts[1])

                instructions[i_instruction] = opcode_exec_instruction(input_register, input_register, input_register, Operation.IO_WRITE)
                continue
                
        except Exception as exception:
            print(f"Exception whilst parsing line {i_src_line}: '{src_line}'")
            raise exception

        raise Exception(f"Failed to parse line {i_src_line}: '{src_line}'")

    memory = np.zeros(PROGRAM_MEMORY_SIZE, dtype=np.uint16)
    memory[0:n_instructions] = instructions[0:n_instructions]

    return AssembledProgram(memory, src_lines, instruction_lines, instruction_mapping, labels)

# MAIN PROGRAM
if __name__ == "__main__":
    # Parse arguments
    argument_parser = argparse.ArgumentParser(
        prog="MCASM",
        description="Assembler for the MCPC",
    )
    argument_parser.add_argument("filename")
    argument_parser.add_argument("-o", "--output")
    arguments = argument_parser.parse_args()
    input_filename: str = arguments.filename
    output_filename: str | None = arguments.output

    # Resolve input and output file path.
    input_filepath = pathlib.Path(input_filename)
    if output_filename is None:
        output_filename = f"{input_filepath.stem}.mcbin"
    output_filepath = pathlib.Path(input_filepath.stem if (output_filename is None) else output_filename)

    # Read input lines
    with open(input_filepath, "r") as input_file:
        src_lines = [line.strip() for line in input_file.readlines()]

    # Assemble the program
    program = assemble(src_lines)

    # Write the output to a file.
    with open(output_filepath, "wb") as output_file:
        program.binary.tofile(output_file, "")

    print(f"Assembled {len(program.instructions)} instructions")
