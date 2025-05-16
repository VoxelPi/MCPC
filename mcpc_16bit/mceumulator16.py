import argparse
import numpy as np
import numpy.typing as npt
import pathlib
import time

from mcpc16 import MEMORY_SIZE, PROGRAM_MEMORY_SIZE, REGISTER_COUNT, Condition, Operation, Register, decode_instruction
import mcasm16

class Emulator:
    program = np.zeros(PROGRAM_MEMORY_SIZE, dtype=np.uint64)
    registers = np.zeros(REGISTER_COUNT, dtype=np.uint16)
    memory = np.zeros(MEMORY_SIZE, dtype=np.uint16)
    halt: bool = False

    @property
    def pc(self) -> np.uint16:
        return self.registers[0]
    
    def register_value(self, register: Register) -> np.uint16:
        return self.registers[register.value]
    
    def set_register_value(self, register: Register, value: np.uint16):
        self.registers[register.value] = value
    
    @pc.setter
    def pc(self, value: np.uint16):
        self.registers[0] = value

    def initialize(self):
        self.registers = np.zeros(REGISTER_COUNT, dtype=np.uint16)
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint16)

    def load_program(self, program: npt.NDArray[np.uint64]):
        np.copyto(self.program, program)
        self.initialize()

    def evaluate_condition(self, condition: Condition, value: np.uint16) -> bool:
        is_zero = (value == 0)
        is_negative = ((value & 0b1000_0000_0000_0000) != 0)

        match condition:
            case Condition.ALWAYS:
                return True
            case Condition.NEVER:
                return False
            case Condition.EQUAL:
                return is_zero
            case Condition.NOT_EQUAL:
                return not is_zero
            case Condition.LESS:
                return is_negative
            case Condition.GREATER_OR_EQUAL:
                return not is_negative
            case Condition.GREATER:
                return (not is_zero) and (not is_negative)
            case Condition.LESS_OR_EQUAL:
                return is_zero or is_negative

    def evaluate_operation(self, operation: Operation, a: np.uint16, b: np.uint16) -> tuple[np.uint16, bool]:
        match operation:
            case Operation.CLEAR:
                return (np.uint16(0), True)
            case Operation.A:
                return (a, True)
            case Operation.AND:
                return (a & b, True)
            case Operation.NAND:
                return (~(a & b), True)
            case Operation.OR:
                return (a | b, True)
            case Operation.NOR:
                return (~(a | b), True)
            case Operation.XOR:
                return (a ^ b, True)
            case Operation.XNOR:
                return (~(a ^ b), True)
            
            case Operation.INC:
                return (a + 1, True)
            case Operation.DEC:
                return (a - 1, True)
            case Operation.ADD:
                return (a + b, True)
            case Operation.SUB:
                return (a - b, True)
            
            case Operation.SHIFT_LEFT:
                return (np.uint16(a << 1), True)
            case Operation.SHIFT_RIGHT:
                return (np.uint16(a >> 1), True)
            case Operation.ROTATE_LEFT:
                return (np.uint16((a << 1) | ((a >> 7) & 0b1)), True)
            case Operation.ROTATE_RIGHT:
                return (np.uint16((a >> 1) | ((a & 0b1) << 7)), True)
            
            case Operation.CHECK_ALWAYS:
                return (np.uint16(self.evaluate_condition(Condition.ALWAYS, a)), True)
            case Operation.CHECK_NEVER:
                return (np.uint16(self.evaluate_condition(Condition.NEVER, a)), True)
            case Operation.CHECK_EQUAL:
                return (np.uint16(self.evaluate_condition(Condition.EQUAL, a)), True)
            case Operation.CHECK_NOT_EQUAL:
                return (np.uint16(self.evaluate_condition(Condition.NOT_EQUAL, a)), True)
            case Operation.CHECK_LESS:
                return (np.uint16(self.evaluate_condition(Condition.LESS, a)), True)
            case Operation.CHECK_GREATER_OR_EQUAL:
                return (np.uint16(self.evaluate_condition(Condition.GREATER_OR_EQUAL, a)), True)
            case Operation.CHECK_GREATER:
                return (np.uint16(self.evaluate_condition(Condition.GREATER, a)), True)
            case Operation.CHECK_LESS_OR_EQUAL:
                return (np.uint16(self.evaluate_condition(Condition.LESS_OR_EQUAL, a)), True)
            
            case Operation.BIT_GET:
                return (np.uint16((a >> b) & 0b1 != 0), True)
            case Operation.BIT_SET:
                return (a | (np.uint16(1) << b), True)
            case Operation.BIT_CLEAR:
                return (a & ~(np.uint16(1) << b), True)
            case Operation.BIT_TOGGEL:
                return (a ^ (np.uint16(1) << b), True)

            case Operation.MEMORY_LOAD:
                return (self.memory[a], True)
            case Operation.MEMORY_STORE:
                self.memory[a] = b
                return (b, False) # Return the value
            
            case Operation.IO_POLL:
                return (np.uint16(1), True) # TODO: Actually check if input is available
            case Operation.IO_READ:
                return (np.uint16(int(input("[IO] Input a number: "), 0)), True)
            case Operation.IO_WRITE:
                print(f"[IO] {a}")
                return (a, False) # Return the value
            
            case Operation.MULTIPLY:
                return (a * b, True)
            case Operation.DIVIDE:
                return (a // b, True)
            case Operation.MODULO:
                return (a % b, True)
            case Operation.SQRT:
                return (np.uint16(int(np.sqrt(a))), True)
            
            case Operation.BREAK:
                self.halt = True
                return (a, False)
            
            case _:
                raise Exception(f"Operation {operation} ({operation.value}) is not implemented")

    def execute_instruction(self):
        # Fetch instruction.
        instruction_opcode = self.program[self.pc]

        # Decode instruction.
        instruction = decode_instruction(instruction_opcode)

        # Execute instruction.
        a_value = self.register_value(instruction.a) if isinstance(instruction.a, Register) else instruction.a
        b_value = self.register_value(instruction.b) if isinstance(instruction.b, Register) else instruction.b

        condition_value = self.register_value(instruction.condition_register)
        condition_valid = self.evaluate_condition(instruction.condition, condition_value)

        operation_result, operation_has_result = self.evaluate_operation(instruction.operation, a_value, b_value)

        # Increment program counter.
        self.pc += 1

        # Store result.
        if condition_valid & operation_has_result:
            self.set_register_value(instruction.output_register, operation_result)

    def execute_instructions(self):
        while not self.halt:
            self.execute_instruction()
        self.halt = False

# MAIN PROGRAM
if __name__ == "__main__":
    # Parse arguments
    argument_parser = argparse.ArgumentParser(
        prog="MCEMULATOR",
        description="Emulator for the MCPC",
    )
    argument_parser.add_argument("filename", nargs="?", default="./mcpc_16bit/programs/calculator.mcasm")
    argument_parser.add_argument("-t", "--time")
    arguments = argument_parser.parse_args()
    input_filename: str = arguments.filename
    clock_time: float = float(arguments.time or 0.0)

    # Read input lines
    input_filepath = pathlib.Path(input_filename)
    if input_filepath.suffix == ".mcasm":
        # Program is assembled source code.
        with open(input_filepath, "r") as input_file:
            src_lines = [line.strip() for line in input_file.readlines()]

        # Assemble the program
        assembled_program = mcasm16.assemble(src_lines)
        print(f"Assembled {len(assembled_program.instructions)} instructions")
        program = assembled_program.binary
    
    else:
        # Program is binary.
        with open(input_filepath, "rb") as input_file:
            program = np.frombuffer(input_file.read(), dtype=np.uint64)
            print(f"Loaded {len(program)} instructions")

    emulator = Emulator()
    emulator.load_program(program)
    try:
        while True:
            emulator.execute_instruction()
            if emulator.halt:
                input("Press any key to continue...")
                emulator.halt = False
            if clock_time > 0:
                time.sleep(clock_time)
    except KeyboardInterrupt:
        exit(0)
