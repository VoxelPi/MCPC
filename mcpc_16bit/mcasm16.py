from __future__ import annotations
import sys
import numpy as np
import numpy.typing as npt
import pathlib
from mcpc16 import Register, Condition, Operation, encode_instruction, PROGRAM_MEMORY_SIZE
import argparse
from dataclasses import dataclass

START_LABEL = "start"

ASSEMBLER_MACROS = {
    "true": "1",
    "false": "0",
    "c1": "r14",
    "c2": "r15",
    "r0": "pc",
}

class AssemblyError(Exception):
    source_line: AssemblySourceLine
    message: str

    def __init__(self, source_line: AssemblySourceLine, message: str):
        super().__init__(f"Failed to parse line {source_line.line + 1} of unit '{source_line.unit}': '{source_line.text}'.\n{message}")
        self.source_line = source_line
        self.message = message

class AssemblySyntaxError(Exception):
    source_line: AssemblySourceLine
    message: str

    def __init__(self, source_line: AssemblySourceLine, message: str):
        super().__init__(f"Failed to parse line {source_line.line + 1} of unit '{source_line.unit}': '{source_line.text}'.\n{message}")
        self.source_line = source_line
        self.message = message

class AssemblyIncludeError(Exception):
    source_line: AssemblySourceLine
    target: str

    def __init__(self, source_line: AssemblySourceLine, target: str):
        super().__init__(f"Failed to resolve included unit '{target}' in line {source_line.line + 1} of unit '{source_line.unit}'.")
        self.source_line = source_line
        self.target = target

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
class AssemblyLabel():
    name: str
    location: int
    source: AssemblySourceLine | None

@dataclass
class AssemblyMacro():
    name: str
    value: str
    source: AssemblySourceLine | None

@dataclass
class AssemblySourceLine():
    unit: str
    line: int
    text: str

@dataclass
class AssemblyInstruction():
    source: AssemblySourceLine
    text: str

@dataclass
class AssembledProgram():
    binary: npt.NDArray[np.uint64]
    text: list[str]
    instructions: list[AssemblyInstruction]
    labels: dict[str, AssemblyLabel]
    macros: dict[str, AssemblyMacro]

    @property
    def n_instructions(self) -> int:
        return len(self.instructions)

def _parse_instruction(instruction_line: AssemblyInstruction) -> np.uint64:
    source_line = instruction_line.source
    instruction_text = instruction_line.text
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
            raise AssemblySyntaxError(source_line, "Invalid condition syntax. Should be 'if <register> <condition> 0'.")
        
        # Get condition.
        if condition_parts[1] not in condition_operators:
            raise AssemblySyntaxError(source_line, f"Invalid condition operator '{condition_parts[1]}'.")
        condition = condition_operators[condition_parts[1]]

        # Get condition source register.
        condition_register = parse_register(condition_parts[0])
        if condition_register is None:
            raise AssemblySyntaxError(source_line, f"Invalid condition source register '{condition_parts[0]}'.")

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
            raise AssemblySyntaxError(source_line, "Invalid syntax for jump. Should be 'jump <register|immediate>'.")

        # Parse the jump target value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid jump target '{instruction_parts[1]}'.")
        
        # Encode instruction.
        return encode_instruction(Operation.A, condition_register, condition, Register.PC, value, 0)
        
    # SKIP
    if instruction_parts[0] == 'skip':
        # Check general syntax.
        if n_instruction_parts != 2:
            raise AssemblySyntaxError(source_line, "Invalid syntax for skip. Should be 'skip <register|immediate>'.")

        # Parse the skip value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid skip length '{instruction_parts[1]}'.")
        
        # Encode instruction.
        return encode_instruction(Operation.ADD, condition_register, condition, Register.PC, value, Register.PC)

    # Assignments, that are instructions of the form '<output_register> = ...'
    if n_instruction_parts >= 2 and is_register(instruction_parts[0]) and instruction_parts[1] == "=":
        # Get the output register
        output_register = parse_register(instruction_parts[0])
        if output_register is None:
            raise AssemblySyntaxError(source_line, f"Invalid register '{instruction_parts[0]}'.")

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
                    raise AssemblySyntaxError(source_line, f"Invalid value for memory address: '{address_text}'.")

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
            raise AssemblySyntaxError(source_line, f"Invalid load value: '{value_text}'.")

        # SHIFT
        # TODO: shift count?
        if n_instruction_parts == 5 and instruction_parts[2] == "shift":
            # Parse value.
            value = parse_value(instruction_parts[4])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[4]}'.")

            # Parse direction.
            match instruction_parts[3]:
                case "left":
                    return encode_instruction(Operation.SHIFT_LEFT, condition_register, condition, output_register, value, 1)
                case "right":
                    return encode_instruction(Operation.SHIFT_RIGHT, condition_register, condition, output_register, value, 1)
                case _:
                    raise AssemblySyntaxError(source_line, f"Invalid shift direction '{instruction_parts[3]}'.")
        
        # ROTATE
        if n_instruction_parts == 5 and instruction_parts[2] == "rotate":
            # Parse value.
            value = parse_value(instruction_parts[4])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[4]}'.")

            # Parse direction.
            match instruction_parts[3]:
                case "left":
                    return encode_instruction(Operation.ROTATE_LEFT, condition_register, condition, output_register, value, 1)
                case "right":
                    return encode_instruction(Operation.ROTATE_RIGHT, condition_register, condition, output_register, value, 1)
                case _:
                    raise AssemblySyntaxError(source_line, f"Invalid rotation direction '{instruction_parts[3]}'.")

        # Bit operation, '<r> = <value> bit <operation> <bit>'
        if n_instruction_parts == 6 and instruction_parts[3] == "bit":
            # Parse value.
            value = parse_value(instruction_parts[2])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[2]}'.")
            
            # Parse bit.
            bit = parse_value(instruction_parts[5])
            if bit is None:
                raise AssemblySyntaxError(source_line, f"Invalid bit '{instruction_parts[5]}'.")
            
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
                    raise AssemblySyntaxError(source_line, f"Invalid bit operation '{bit_operation}'.")

        # Unary operators
        if n_instruction_parts == 4 and instruction_parts[2] in unary_operators:
            # Parse value.
            value = parse_value(instruction_parts[3])
            if value is None:
                raise AssemblySyntaxError(source_line, f"Invalid value '{instruction_parts[3]}'.")
            
            # Parse operation.
            operation = unary_operators[instruction_parts[2]]

            # Encode instruction.
            return encode_instruction(operation, condition_register, condition, output_register, value, 0)

        # Binary operators
        if n_instruction_parts == 5 and instruction_parts[3] in binary_operations:
            # Parse value for A.
            value_a = parse_value(instruction_parts[2])
            if value_a is None:
                raise AssemblySyntaxError(source_line, f"Invalid value for A: '{instruction_parts[3]}'.")
            
            # Parse value for B.
            value_b = parse_value(instruction_parts[4])
            if value_b is None:
                raise AssemblySyntaxError(source_line, f"Invalid value for A: '{instruction_parts[3]}'.")

            # Parse operation.
            operation = binary_operations[instruction_parts[3]]

            # Encode instruction.
            return encode_instruction(operation, condition_register, condition, output_register, value_a, value_b)

        raise AssemblySyntaxError(source_line, "Invalid right hand side for assignment")

    # Memory store, '[<address>] = <value>'
    if n_instruction_parts == 3 and instruction_parts[0][0] == "[" and instruction_parts[0][-1] == "]" and instruction_parts[1] == "=":
        # Parse address.
        address_text = instruction_parts[0][1:-1]
        address = parse_value(address_text)
        if address is None:
            raise AssemblySyntaxError(source_line, f"Invalid memory address: '{address_text}'.")
        
        # Parse value.
        value = parse_value(instruction_parts[2])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{instruction_parts[2]}'.")
    
        # Encode instruction.
        return encode_instruction(Operation.MEMORY_STORE, condition_register, Condition.NEVER, Register.R1, address, value)

    # Stack push, 'push <value>'
    if n_instruction_parts == 2 and instruction_parts[0] == "push":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{value}'.")

        # Encode instruction.
        return encode_instruction(Operation.STACK_PUSH, condition_register, Condition.NEVER, Register.R1, value, 0)

    # Call, 'call <address>'
    if n_instruction_parts == 2 and instruction_parts[0] == "call":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{value}'.")

        # Encode instruction.
        return encode_instruction(Operation.STACK_CALL, condition_register, condition, Register.PC, Register.PC, value)

    # IO WRITE, 'write <value>'
    if n_instruction_parts == 2 and instruction_parts[0] == "write":
        # Parse value.
        value = parse_value(instruction_parts[1])
        if value is None:
            raise AssemblySyntaxError(source_line, f"Invalid value: '{value}'.")

        # Encode instruction.
        return encode_instruction(Operation.IO_WRITE, condition_register, Condition.NEVER, Register.R1, value, 0)
    
    # No instruction pattern detected, throw generic syntax exception.
    raise AssemblySyntaxError(source_line, f"Invalid instruction: '{instruction_text}'.")

def _preformat_source_line(line: str) -> str:
    # Remove comments
    line = line.split('#', 1)[0]

    # Strip line
    line = line.strip()

    # Simplify whitespace.
    line = line.replace("    ", " ")
    line = " ".join(line.split())

    # Convert to lowercase.
    line = line.lower()

    return line

def _prepare_instruction_lines(
    src_lines: list[str],
    unit: str,
    include_directories: list[pathlib.Path],
    include_scope: set[str] | None = None,
) -> list[AssemblyInstruction]:
    # Early exit if program is empty.
    if len(src_lines) == 0:
        return []
    
    # Prepare include scope.
    if include_scope is None:
        include_scope = set()
    include_scope = include_scope | { unit }
    
    # Create the initial source mapping.
    instruction_lines: list[AssemblyInstruction] = [ AssemblyInstruction(AssemblySourceLine(unit, line, text.strip()), _preformat_source_line(text)) for (line, text) in enumerate(src_lines) ]

    # Remove empty lines.
    i_line = 0
    while i_line < len(instruction_lines):
        if instruction_lines[i_line].text == "":
            # Delete line, DON'T increment line counter.
            del instruction_lines[i_line]
        else:
            # Increment line counter.
            i_line += 1

    # Handle includes.
    i_line = 0
    while i_line < len(instruction_lines):
        line_text = instruction_lines[i_line].text
        if not line_text.startswith("!include "):
            i_line += 1
            continue

        # Include statement.
        include_target_id = line_text[9:]
        include_file_path: pathlib.Path | None = None
        for include_directory in include_directories:
            file_path = include_directory / f"{include_target_id}.mcasm"
            if file_path.exists() and file_path.is_file():
                include_file_path = file_path
                break

        # Raise exception if include target cannot be resolved.
        if include_file_path is None:
            raise AssemblyIncludeError(instruction_lines[i_line].source, include_target_id)
        
        # Read include file
        try:
            with open(include_file_path, "r") as include_file:
                include_src_lines = include_file.readlines()
        except Exception:
            raise AssemblyIncludeError(instruction_lines[i_line].source, include_target_id) 

        # Prepare include lines recursivly.
        include_instruction_lines = _prepare_instruction_lines(include_src_lines, str(include_file_path), include_directories, include_scope)

        # Replace include statement with included lines.
        instruction_lines = instruction_lines[:i_line] + include_instruction_lines + instruction_lines[(i_line+1):]

        # Skip included lines as they have already been processed.
        i_line += len(include_instruction_lines)

    return instruction_lines

def assemble(
    src_lines: list[str],
    unit: str,
    include_directories: list[pathlib.Path],
    default_macros: dict[str, str] = {},
    default_labels: dict[str, int] = {},
) -> AssembledProgram | None:
    # Early exit if program is empty.
    if len(src_lines) == 0:
        return AssembledProgram(np.zeros(1, dtype=np.uint64), [], [], {}, {})

    # Prepare the source lines.
    # This handles things like comments, whitespace simplifcation and lowercase transforamtion, 
    # as well as the !include statement.
    instruction_lines = _prepare_instruction_lines(src_lines, str(unit), include_directories)

    # Preprocessor
    # Handles labels, macros
    labels: dict[str, AssemblyLabel] = { name : AssemblyLabel(name, location, None) for (name, location) in default_labels.items() }
    macros: dict[str, AssemblyMacro] = { name : AssemblyMacro(name, value, None) for (name, value) in (ASSEMBLER_MACROS | default_macros).items() }
    i_instruction = 0
    while i_instruction < len(instruction_lines):
        instruction = instruction_lines[i_instruction]
        
        # Handle labels.
        if instruction.text.startswith("@"):
            # Parse label.
            label_name = instruction.text[1:]

            # Check that label is unique.
            existing_label = labels.get(label_name)
            if existing_label is not None:
                existing_source = existing_label.source
                if existing_source is not None:
                    raise AssemblyError(instruction.source, f"Trying to redefine label '{label_name}' which is already defined in line {existing_source.line} of unit '{existing_source.unit}'.")
                else:
                    raise AssemblyError(instruction.source, f"Trying to redefine label '{label_name}' which is already defined in the default labels.")

            # Save label.
            labels[label_name] = AssemblyLabel(label_name, i_instruction, instruction.source)

            # Delete original instruction line, DON'T increment line counter.
            del instruction_lines[i_instruction]
            continue

        # Handle macros.
        if instruction.text.startswith("!define"):
            # Parse macro.
            define_parts = instruction.text.split(" ", 2)
            if len(define_parts) < 3:
                raise AssemblySyntaxError(instruction.source, "Invalid macro definition. Syntax is '!define <name> <value>'.")
            macro_name = define_parts[1]
            macro_value = " ".join(define_parts[2:])

            # Check that label is unique.
            existing_macro = macros.get(macro_name)
            if existing_macro is not None:
                existing_source = existing_macro.source
                if existing_source is not None:
                    raise AssemblyError(instruction.source, f"Trying to redefine macro '{macro_name}' which is already defined in line {existing_source.line} of unit '{existing_source.unit}'.")
                else:
                    raise AssemblyError(instruction.source, f"Trying to redefine macro '{macro_name}' which is already defined in the default macros.")

            # Save macro.
            macros[macro_name] = AssemblyMacro(macro_name, macro_value, instruction.source)

            # Delete original instruction line, DON'T increment line counter.
            del instruction_lines[i_instruction]
            continue

        # Increment line counter.
        i_instruction += 1

    # Check if the start label was used.
    if START_LABEL in labels:
        start_label = labels[START_LABEL]
        i_start_instruction = start_label.location
        if i_start_instruction != 0:
            # Insert an artificial "jump @start" instruction as instruction 0.
            instruction_lines.insert(0, AssemblyInstruction(AssemblySourceLine("__generated__", 0, "jump @start"), f"jump {i_start_instruction + 1}"))

            # Increase all label locations, as a new instruction has been added.
            for label in labels.values():
                label.location += 1

    # Count instructions. The number of instruction doesn't change after this point.
    n_instructions = len(instruction_lines)

    # Early exit if program is empty.
    if n_instructions == 0:
        return AssembledProgram(np.zeros(1, dtype=np.uint64), src_lines, instruction_lines, labels, macros)

    # Apply labels and macros.
    for instruction in instruction_lines:
        # Add whitespace to fix word replace at line start / end
        instruction.text = f" {instruction.text} "

        # Apply labels.
        for label in labels.values():
            instruction.text = instruction.text.replace(f" @{label.name} ", f" {label.location} ")
            instruction.text = instruction.text.replace(f"[@{label.name}]", f"[{label.location}]")

        # Apply macros.
        for macro in macros.values():
            instruction.text = instruction.text.replace(f" {macro.name} ", f" {macro.value} ")
            instruction.text = instruction.text.replace(f"[{macro.name}]", f"[{macro.value}]")
            
        # Remove previously added whitespace.
        instruction.text = instruction.text[1:-1]

    # Assemble instructions.
    instructions = np.zeros(n_instructions, dtype=np.uint64)
    for i_instruction, instruction_line in enumerate(instruction_lines):
        try:
            # Parse the instruction.
            instructions[i_instruction] = _parse_instruction(instruction_line)
        except AssemblySyntaxError:
            raise
        except Exception as exception:
            raise Exception(f"Exception whilst parsing line {instruction_line.source.line + 1} of unit '{instruction_line.source.unit}': '{instruction_line.source.text}'.", exception)

    # Return generated program.
    return AssembledProgram(instructions, src_lines, instruction_lines, labels, macros)

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
    
    argument_parser.add_argument("filename", nargs="?", default="./mcpc_16bit/programs/stdlib_test.mcasm")
    argument_parser.add_argument("-o", "--output")
    argument_parser.add_argument("-c", "--check", action="store_true")
    argument_parser.add_argument("-m", "--mappings", action="store_true")
    arguments = argument_parser.parse_args()
    input_filename: str = arguments.filename
    output_filename: str | None = arguments.output
    check_mode: bool = arguments.check
    generate_mappings: bool = arguments.mappings

    # Resolve input and output file path.
    input_filepath = pathlib.Path(input_filename)
    if output_filename is None:
        output_filename = f"{input_filepath.stem}.mcbin"
    output_filepath = pathlib.Path(output_filename)

    # Prepare include paths.
    include_paths: list[pathlib.Path] = [
        pathlib.Path.cwd() / "stdlib",
    ]
    include_paths = [ path for path in include_paths if path.exists() and path.is_dir() ]

    # Read input lines
    with open(input_filepath, "r") as input_file:
        src_lines = input_file.readlines()

    # Assemble the program
    program = None
    try:
        program = assemble(src_lines, str(input_filepath.absolute()), include_paths)
    except AssemblySyntaxError as exception:
        print(exception, file=sys.stderr)
        exit(1)
    except AssemblyIncludeError as exception:
        print(exception, file=sys.stderr)
        exit(1)

    if program is None:
        print("Failed to assemble program.", file=sys.stderr)
        exit(1)

    # Write the output to a file.
    if not check_mode:
        with open(output_filepath, "wb") as output_file:
            output_file.write(program.binary.tobytes())

    # Generate mappings if enabled.
    if generate_mappings:
        mappings_path = output_filepath.parent / f"{output_filepath.stem}.mcmap"
        with open(mappings_path, "w") as mappings_file:
            max_instruction_source_length = np.max([len(line.source.text) for line in program.instructions])

            for instruction in program.instructions:
                mappings_file.write(f"{instruction.source.text.ljust(max_instruction_source_length + 3)} # line {instruction.source.line + 1:5d} of unit \"{instruction.source.unit}\"\n")

    # Summary output.
    print(f"Assembled {len(program.instructions)} instructions")
