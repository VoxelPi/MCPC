import numpy as np
import numpy.typing as npt
import pathlib
from mcpc8 import PROGRAM_MEMORY_SIZE, Register, Condition, Operation, encode_operation_instruction, encode_load_instruction, operation_check
import argparse
from dataclasses import dataclass

ASSEMBLER_SYMBOLS = {
    "true": "1",
    "false": "0",
    "c1": "r7"
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
    if len(text) != 2:
        return None
    
    text = text.upper()
    if text == "PC":
        return Register.PC
    if text[0] == "R" and text[1].isdigit() and (int(text[1]) in range(8)):
        return list(Register)[int(text[1])]
    return None
    
def is_register(text: str) -> bool:
    if len(text) != 2:
        return False
    
    text = text.upper()
    if text == "PC":
        return True
    if text[0] == "R" and text[1].isdigit() and (int(text[1]) in range(8)):
        return True
    return False

def parse_immediate_value(text: str) -> np.uint8 | None:
    try:
        return np.uint8(int(text, 0) & 0xFF)
    except ValueError:
        return None

def is_immediate_value(text: str) -> bool:
    try:
        _ = int(text, 0)
        return True
    except ValueError:
        return False
    
def parse_value(text: str) -> Register | np.uint8 | None:
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
    binary: npt.NDArray[np.uint16]
    text: list[str]
    instructions: list[str]
    src_mapping: list[int] # Instruction id to code line.
    labels: dict[str, int]

    @property
    def n_instructions(self) -> int:
        return len(self.instructions)
    
def _parse_instruction(instruction_text: str, i_instruction: int, source_line: str, i_source_line: int) -> np.uint16:
    instruction_parts = instruction_text.split(" ")
    n_instruction_parts = len(instruction_parts)

    # NO ARGS
    if instruction_text in no_args_instructions:
        operation = no_args_instructions[instruction_text]
        return encode_operation_instruction(Register.R1, Register.R1, Register.R1, operation)

    # JUMP
    if instruction_parts[0] == 'jump':

        # UNCONDITONAL JUMP
        if n_instruction_parts == 2:
            # Parse jump target.
            jump_target = parse_value(instruction_parts[1])
            if jump_target is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid jump target '{instruction_parts[1]}'.")

            # Jump target is a register.
            if isinstance(jump_target, Register):
                return encode_operation_instruction(jump_target, jump_target, Register.PC, Operation.AND)
            else:
                return encode_load_instruction(False, Condition.ALWAYS, Register.PC, jump_target)

        # CONDITIONAL JUMP
        if n_instruction_parts >= 3 and instruction_parts[2] == "if":
            # Check general condition syntax.
            if n_instruction_parts != 6 or instruction_parts[5] != "0":
                raise AssemblySyntaxError(i_source_line, source_line, "Invalid condition syntax. Should be 'jump <address> if <register> <condition> 0'.")
            
            # Parse condition.
            if instruction_parts[4] not in condition_operators:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition operator '{instruction_parts[4]}'.")
            condition = condition_operators[instruction_parts[4]]
            
            # Check that R7 is used as condition value source.
            if instruction_parts[3] != "r7":
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition value source register '{instruction_parts[3]}'.")

            # Parse jump target immediate value.
            jump_target = parse_immediate_value(instruction_parts[1])
            if jump_target is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid jump target '{instruction_parts[1]}'.")
            
            # Parse condition
            return encode_load_instruction(False, condition, Register.PC, jump_target)

        # Invalid jump sytnax.
        raise AssemblySyntaxError(i_source_line, source_line, "Invalid syntax for jump. Should be 'jump <address> [if <register> <condition> 0]'.")
    
    # SKIP
    if instruction_parts[0] == 'skip':

        # UNCONDITONAL SKIP
        if n_instruction_parts == 2:
            # Parse skip length.
            skip_length = parse_value(instruction_parts[1])
            if skip_length is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid skip length '{instruction_parts[1]}'.")

            if isinstance(skip_length, Register):
                # Skip length is a register.
                return encode_operation_instruction(Register.PC, skip_length, Register.PC, Operation.ADD)
            else:
                # Skip length is an immediate value.
                return encode_load_instruction(True, Condition.ALWAYS, Register.PC, skip_length)

        # CONDITIONAL SKIP
        if n_instruction_parts >= 3 and instruction_parts[2] == "if":
            # Check general condition syntax.
            if n_instruction_parts != 6 or instruction_parts[5] != "0":
                raise AssemblySyntaxError(i_source_line, source_line, "Invalid condition syntax. Should be 'skip <value> if <register> <condition> 0'.")

            # Parse condition.
            if instruction_parts[4] not in condition_operators:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition operator '{instruction_parts[4]}'.")
            condition = condition_operators[instruction_parts[4]]
            
            # Check that R7 is used as condition value source.
            if instruction_parts[3] != "r7":
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition value source register '{instruction_parts[3]}'.")

            # Parse skip length immediate value.
            skip_length = parse_immediate_value(instruction_parts[1])
            if skip_length is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid skip length '{instruction_parts[1]}'.")
            
            return encode_load_instruction(True, condition, Register.PC, skip_length)

        # Invalid skip sytnax.
        raise Exception(i_source_line, source_line, "Invalid syntax for skip. Should be 'skip <address> [if <register> <condition> 0]'.")
    
    # RELATIVE LOADS
    if n_instruction_parts >= 2 and instruction_parts[1] == "+=":
        # Parse the output register
        output_register = parse_register(instruction_parts[0])
        if output_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register '{instruction_parts[0]}'.")

        # RELATIVE LOAD
        if n_instruction_parts == 3:
            # Parse value
            value = parse_value(instruction_parts[2])
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid value '{instruction_parts[2]}'.")

            if isinstance(value, Register):
                # Value is a register.
                return encode_operation_instruction(output_register, value, output_register, Operation.ADD)
            else:
                # Value is an immediate value.
                return encode_load_instruction(True, Condition.ALWAYS, output_register, value)

        # RELATIVE CONDITIONAL LOAD
        if n_instruction_parts >= 4 and instruction_parts[3] == "if":
            # Check general condition syntax.
            if n_instruction_parts != 7 or instruction_parts[6] != "0":
                raise AssemblySyntaxError(i_source_line, source_line, "Invalid syntax. Should be '<register> += <value> if <register> <condition> 0'.")

            # Parse condition.
            if instruction_parts[5] not in condition_operators:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition operator '{instruction_parts[5]}'.")
            condition = condition_operators[instruction_parts[5]]

            # Check that R7 is used as condition value source.
            if instruction_parts[4] != "r7":
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition value source register '{instruction_parts[3]}'.")
            
            # Parse immediate value.
            value = parse_immediate_value(instruction_parts[2])
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid immediate value '{instruction_parts[2]}'.")
        
            return encode_load_instruction(True, condition, output_register, value)

    # MEMORY STORE
    if n_instruction_parts == 3 and instruction_parts[0][0] == "[" and instruction_parts[0][-1] == "]" and instruction_parts[1] == "=":
        # Parse address.
        address_text = instruction_parts[0][1:-1]
        address_register = parse_register(address_text)
        if address_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register for memroy address '{address_text}'") 

        # Parse value.
        input_register = parse_register(instruction_parts[2])
        if input_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register for memory value: '{instruction_parts[2]}'.")
    
        # Encode instruction.
        return encode_operation_instruction(address_register, input_register, input_register, Operation.MEMORY_STORE)

    # Assignments
    if n_instruction_parts >= 2 and instruction_parts[1] == "=":
        # Get the output register
        output_register = parse_register(instruction_parts[0])
        if output_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register '{instruction_parts[0]}'.")

        # LOAD
        if n_instruction_parts == 3:
            value_text = instruction_parts[2]

            # RANDOM VALUE
            if value_text == "random":
                return encode_operation_instruction(Register.R1, Register.R1, output_register, Operation.RANDOM)

            # FROM IO POLL
            if value_text == "poll":
                return encode_operation_instruction(Register.R1, Register.R1, output_register, Operation.IO_POLL)

            # FROM IO READ
            if value_text == "read":
                return encode_operation_instruction(Register.R1, Register.R1, output_register, Operation.IO_READ)

            # FROM MEMORY
            if value_text[0] == "[" and value_text[-1] == "]":
                # Parse the address
                address_text = value_text[1:-1]
                address_register = parse_register(address_text)
                if address_register is None:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register for memory address: '{address_text}'.")

                # Encode instruction.
                return encode_operation_instruction(address_register, address_register, output_register, Operation.MEMORY_LOAD)

            # A VALUE
            value = parse_value(value_text)
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid syntax for load value: '{value_text}'")
            
            if isinstance(value, Register):
                # Value is a register.
                return encode_operation_instruction(value, value, output_register, Operation.AND)
            else:
                # Value is an immediate value.
                return encode_load_instruction(False, Condition.ALWAYS, output_register, value)

        # CONDITIONAL LOAD
        if n_instruction_parts >= 4 and instruction_parts[3] == "if" and instruction_parts[5] in condition_operators:
            # Check general condition syntax.
            if n_instruction_parts != 7 or instruction_parts[6] != "0":
                raise AssemblySyntaxError(i_source_line, source_line, "Invalid condition syntax. Should be '<register> = <value> if <register> <condition> 0'.")
            
            # Parse condition.
            if instruction_parts[5] not in condition_operators:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition operator '{instruction_parts[5]}'.")
            condition = condition_operators[instruction_parts[5]]
            
            # Check that R7 is used as condition value source.
            if instruction_parts[4] != "r7":
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition value source register '{instruction_parts[4]}'.")
            
            # Parse jump target immediate value.
            value = parse_immediate_value(instruction_parts[2])
            if value is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid immediate value '{instruction_parts[2]}'.")
        
            # Encode instruction.
            return encode_load_instruction(False, condition, output_register, value)

        # BIT GET
        if n_instruction_parts == 5 and instruction_parts[3] == "bit":
            # Parse input register.
            input_register = parse_register(instruction_parts[2])
            if input_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register '{instruction_parts[2]}'.")

            # Parse bit index.
            bit_index = parse_immediate_value(instruction_parts[4])
            if bit_index is None or bit_index < 0 or bit_index > 7:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid bit index {bit_index}")
            operation = list(Operation)[int(Operation.BIT_GET_0.value + bit_index)]
            
            # Encode instruction.
            return encode_operation_instruction(input_register, input_register, output_register, operation)

        # SHIFT <r> = shift right <ri>
        if n_instruction_parts == 5 and instruction_parts[2] == "shift":
            # Parse input register.
            input_register = parse_register(instruction_parts[4])
            if input_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register '{instruction_parts[4]}'.")

            # Parse direction
            direction = instruction_parts[3]
            match direction:
                case "left":
                    return encode_operation_instruction(input_register, input_register, output_register, Operation.SHIFT_LEFT)
                case "right":
                    return encode_operation_instruction(input_register, input_register, output_register, Operation.SHIFT_RIGHT)
                case _:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid shift direction '{instruction_parts[3]}'.")
        
        # ROTATE
        if n_instruction_parts == 5 and instruction_parts[2] == "rotate":
            # Parse input register.
            input_register = parse_register(instruction_parts[4])
            if input_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register '{instruction_parts[4]}'.")

            # Parse direction
            direction = instruction_parts[3]
            match direction:
                case "left":
                    return encode_operation_instruction(input_register, input_register, output_register, Operation.ROTATE_LEFT)
                case "right":
                    return encode_operation_instruction(input_register, input_register, output_register, Operation.ROTATE_RIGHT)
                case _:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid shift direction '{instruction_parts[3]}'.")

        # CHECK
        if n_instruction_parts >= 3 and instruction_parts[2] == "check":
            # Check general condition syntax.
            if instruction_parts[5] != "0" or n_instruction_parts != 6:
                raise AssemblySyntaxError(i_source_line, source_line, "Invalid condition syntax for check")

            # Parse input.
            input_register = parse_register(instruction_parts[3])
            if input_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register: '{instruction_parts[3]}'.") 

            # Parse condition.
            if instruction_parts[4] not in condition_operators:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid condition '{instruction_parts[4]}'.")
            condition = condition_operators[instruction_parts[4]]
            operation = operation_check(condition)

            # Encode instruction.
            return encode_operation_instruction(input_register, input_register, output_register, operation)

        # MAC
        if n_instruction_parts == 5 and instruction_parts[2] == "mac":
            # Parse A.
            input_a_register = parse_register(instruction_parts[3])
            if input_a_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register for a '{instruction_parts[3]}'.")
            
            # Parse B.
            input_b_register = parse_register(instruction_parts[4])
            if input_b_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register for b '{instruction_parts[4]}'.")
            
            return encode_operation_instruction(input_a_register, input_b_register, output_register, Operation.MULTIPLY_ACCUMULATE)

        # MAC RS
        if n_instruction_parts == 5 and instruction_parts[2] == "macrs" and instruction_parts[3] == "adc":
             # Parse input.
            input_register = parse_register(instruction_parts[4])
            if input_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register '{instruction_parts[4]}'.")
            
            return encode_operation_instruction(input_register, input_register, output_register, Operation.MULTIPLY_ACCUMULATE_RESET)

        # UNARY OPERATIONS '<out> = <op> <a>'.
        if n_instruction_parts == 4 and instruction_parts[2] in unary_operators:
            # Parse input.
            input_register = parse_register(instruction_parts[3])
            if input_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register '{instruction_parts[3]}'.")

            # Parse operation.
            operation = unary_operators[instruction_parts[2]]

            # Encode instruction.
            return encode_operation_instruction(input_register, input_register, output_register, operation)

        # BINARY OPERATIONS '<out> = <a> <op> <b>'
        if n_instruction_parts == 5 and instruction_parts[3] in binary_operations:
            # Parse A.
            input_a_register = parse_register(instruction_parts[2])
            if input_a_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register for a '{instruction_parts[2]}'.")
            
            # Parse B.
            input_b_register = parse_register(instruction_parts[4])
            if input_b_register is None:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register for b '{instruction_parts[4]}'.")

            # Parse operation.
            operation = binary_operations[instruction_parts[3]]

            return encode_operation_instruction(input_a_register, input_b_register, output_register, operation)

        raise AssemblySyntaxError(i_source_line, source_line, "Invalid right hand side for assignment")

    # BIT SET
    if n_instruction_parts == 5 and instruction_parts[1] == "bit" and instruction_parts[3] == "=":
        # Parse output register.
        output_register = parse_register(instruction_parts[0])
        if output_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid output register '{instruction_parts[0]}'.")
        
        # Parse input register.
        input_register = parse_register(instruction_parts[4])
        if input_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid input register '{instruction_parts[4]}'.")
        
        # Parse bit index.
        bit_index = parse_immediate_value(instruction_parts[2])
        if bit_index is None or bit_index < 0 or bit_index > 7:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid bit index '{instruction_parts[2]}'")
        operation = list(Operation)[int(Operation.BIT_SET_0.value + bit_index)]
        
        # Encode instruction.
        return encode_operation_instruction(input_register, output_register, output_register, operation)

    # CLEAR
    if instruction_parts[0] == "clear":
        # Check general syntax.
        if len(instruction_parts) != 2:
            raise AssemblySyntaxError(i_source_line, source_line, "Invalid arguments, syntax: 'clear <register>'")
        
        # Parse output register.
        output_register = parse_register(instruction_parts[1])
        if output_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register: '{instruction_parts[1]}'.")
        
        # Encode instruction.
        return encode_operation_instruction(Register.R1, Register.R1, output_register, Operation.CLEAR)

    # FILL
    if instruction_parts[0] == "fill":
        # Check general syntax.
        if len(instruction_parts) != 2:
            raise AssemblySyntaxError(i_source_line, source_line, "Invalid arguments, syntax: 'fill <register>'")
        
        # Parse output register.
        output_register = parse_register(instruction_parts[1])
        if output_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register: '{instruction_parts[1]}'")
        
        # Encode instruction.
        return encode_operation_instruction(Register.R1, Register.R1, output_register, Operation.FILL)

    # CONFIGURE
    if instruction_parts[0] == "configure":
        # Parse configuration target.
        if len(instruction_parts) < 2:
            raise AssemblySyntaxError(i_source_line, source_line, "Invalid arguments, syntax: 'configure <target>'")
        configuration_target = instruction_parts[1]
        
        match configuration_target:
            case "dac":
                # Check syntax.
                if len(instruction_parts) < 3 or not is_register(instruction_parts[2]):
                    raise AssemblySyntaxError(i_source_line, source_line, "DAC configuration requires one input.")
                
                # Parse input reigster.
                input_register = parse_register(instruction_parts[2])
                if input_register is None:
                    raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register: '{instruction_parts[2]}'")
                
                # Encode instruction.
                return encode_operation_instruction(input_register, input_register, input_register, Operation.CONFIGURE_DAC)
                
            case _:
                raise AssemblySyntaxError(i_source_line, source_line, f"Invalid configuration target: '{configuration_target}'")

    # IO WRITE
    if n_instruction_parts == 2 and instruction_parts[0] == "write":
        # Parse input reigster.
        input_register = parse_register(instruction_parts[1])
        if input_register is None:
            raise AssemblySyntaxError(i_source_line, source_line, f"Invalid register: '{instruction_parts[1]}'")

        # Encode instruction.
        return encode_operation_instruction(input_register, input_register, input_register, Operation.IO_WRITE)
    
    # No instruction pattern detected, throw generic syntax exception.
    raise AssemblySyntaxError(i_source_line, source_line, "Invalid instruction")

def assemble(src_lines: list[str], default_macro_symbols: dict[str, str] = {}) -> AssembledProgram:
    # Remove comments
    code_lines = np.array([line.split('#', 1)[0].strip() for line in src_lines])

    if len(code_lines) == 0:
        return AssembledProgram(np.zeros(256, dtype=np.uint16), src_lines, code_lines.tolist(), [], dict())

    # Remove empty lines
    src_to_code_line = np.roll(np.cumsum(code_lines != ""), 1)
    src_to_code_line[0] = 0
    code_line_mapping: list[int] = np.zeros(np.max(src_to_code_line) + 1).tolist()
    for i, line in enumerate(src_to_code_line):
        code_line_mapping[line] = i
    code_lines = code_lines[code_line_mapping]

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
        try:
            # Parse the instruction
            instructions[i_instruction] = _parse_instruction(instruction_text, i_instruction, src_line, i_src_line)
    
        except Exception as exception:
            print(f"Exception whilst parsing line {i_src_line}: '{src_line}'")
            raise exception

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
    output_filepath = pathlib.Path(output_filename)

    # Read input lines
    with open(input_filepath, "r") as input_file:
        src_lines = [line.strip() for line in input_file.readlines()]

    # Assemble the program
    program = assemble(src_lines)

    # Write the output to a file.
    with open(output_filepath, "wb") as output_file:
        program.binary.tofile(output_file, "")

    print(f"Assembled {len(program.instructions)} instructions")
