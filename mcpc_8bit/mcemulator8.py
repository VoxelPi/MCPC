import argparse
import numpy as np
import numpy.typing as npt
import pathlib
import time

from mcpc8 import PROGRAM_MEMORY_SIZE, REGISTER_COUNT, MEMORY_SIZE, Condition, Register, Operation
import mcasm8 as mcasm8

class Emulator:
    program = np.zeros(PROGRAM_MEMORY_SIZE, dtype=np.uint16)
    registers = np.zeros(REGISTER_COUNT, dtype=np.uint8)
    memory = np.zeros(MEMORY_SIZE, dtype=np.uint8)
    accumulator = np.uint16(0)
    halt: bool = False

    @property
    def pc(self) -> np.uint8:
        return self.registers[0]
    
    @pc.setter
    def pc(self, value: np.uint8):
        self.registers[0] = value
    
    def register_value(self, register: Register) -> np.uint8:
        return self.registers[register.value]
    
    def set_register_value(self, register: Register, value: np.uint8):
        self.registers[register.value] = value

    def initialize(self):
        self.registers = np.zeros(REGISTER_COUNT, dtype=np.uint8)
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint8)

    def load_program(self, program: npt.NDArray[np.uint16]):
        np.copyto(self.program, program)
        self.initialize()

    def evaluate_condition(self, condition: Condition, value: np.uint8) -> bool:
        is_zero = (value == 0)
        is_negative = ((value & 0b1000_0000) != 0)

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

    def evaluate_operation(self, operation: Operation, a: np.uint8, b: np.uint8) -> tuple[np.uint8, bool]:
        match operation:
            case Operation.CLEAR:
                return (np.uint8(0), True)
            case Operation.FILL:
                return (np.uint8(0xFF), True)
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
                return (np.uint8(a << 1), True)
            case Operation.SHIFT_RIGHT:
                return (np.uint8(a >> 1), True)
            case Operation.ROTATE_LEFT:
                return (np.uint8((a << 1) | ((a >> 7) & 0b1)), True)
            case Operation.ROTATE_RIGHT:
                return (np.uint8((a >> 1) | ((a & 0b1) << 7)), True)
            
            case Operation.CHECK_ALWAYS:
                return (np.uint8(self.evaluate_condition(Condition.ALWAYS, a)), True)
            case Operation.CHECK_NEVER:
                return (np.uint8(self.evaluate_condition(Condition.NEVER, a)), True)
            case Operation.CHECK_EQUAL:
                return (np.uint8(self.evaluate_condition(Condition.EQUAL, a)), True)
            case Operation.CHECK_NOT_EQUAL:
                return (np.uint8(self.evaluate_condition(Condition.NOT_EQUAL, a)), True)
            case Operation.CHECK_LESS:
                return (np.uint8(self.evaluate_condition(Condition.LESS, a)), True)
            case Operation.CHECK_GREATER_OR_EQUAL:
                return (np.uint8(self.evaluate_condition(Condition.GREATER_OR_EQUAL, a)), True)
            case Operation.CHECK_GREATER:
                return (np.uint8(self.evaluate_condition(Condition.GREATER, a)), True)
            case Operation.CHECK_LESS_OR_EQUAL:
                return (np.uint8(self.evaluate_condition(Condition.LESS_OR_EQUAL, a)), True)
            
            case Operation.BIT_GET_0:
                return (np.uint8((a >> 0) & 0b1), True)
            case Operation.BIT_GET_1:
                return (np.uint8((a >> 1) & 0b1), True)
            case Operation.BIT_GET_2:
                return (np.uint8((a >> 2) & 0b1), True)
            case Operation.BIT_GET_3:
                return (np.uint8((a >> 3) & 0b1), True)
            case Operation.BIT_GET_4:
                return (np.uint8((a >> 4) & 0b1), True)
            case Operation.BIT_GET_5:
                return (np.uint8((a >> 5) & 0b1), True)
            case Operation.BIT_GET_6:
                return (np.uint8((a >> 6) & 0b1), True)
            case Operation.BIT_GET_7:
                return (np.uint8((a >> 7) & 0b1), True)
            
            case Operation.BIT_SET_0:
                return (np.uint8((b & ~(np.uint8(1) << 0)) | (np.uint8(0 if (a == 0) else 1) << 0)), True)
            case Operation.BIT_SET_1:
                return (np.uint8((b & ~(np.uint8(1) << 1)) | (np.uint8(0 if (a == 0) else 1) << 1)), True)
            case Operation.BIT_SET_2:
                return (np.uint8((b & ~(np.uint8(1) << 2)) | (np.uint8(0 if (a == 0) else 1) << 2)), True)
            case Operation.BIT_SET_3:
                return (np.uint8((b & ~(np.uint8(1) << 3)) | (np.uint8(0 if (a == 0) else 1) << 3)), True)
            case Operation.BIT_SET_4:
                return (np.uint8((b & ~(np.uint8(1) << 4)) | (np.uint8(0 if (a == 0) else 1) << 4)), True)
            case Operation.BIT_SET_5:
                return (np.uint8((b & ~(np.uint8(1) << 5)) | (np.uint8(0 if (a == 0) else 1) << 5)), True)
            case Operation.BIT_SET_6:
                return (np.uint8((b & ~(np.uint8(1) << 6)) | (np.uint8(0 if (a == 0) else 1) << 6)), True)
            case Operation.BIT_SET_7:
                return (np.uint8((b & ~(np.uint8(1) << 7)) | (np.uint8(0 if (a == 0) else 1) << 7)), True)

            case Operation.MEMORY_LOAD:
                return (self.memory[a], True)
            case Operation.MEMORY_STORE:
                self.memory[a] = b
                return (b, False) # Return the value
            
            case Operation.IO_POLL:
                return (np.uint8(1), True) # TODO: Actually check if input is available
            case Operation.IO_READ:
                return (np.uint8(int(input("[IO] Input a number: "), 0)), True)
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
                return (np.uint8(np.sqrt(a)), True)
            
            case Operation.MULTIPLY_ACCUMULATE:
                self.accumulator += np.uint8(a * b)
                return (np.uint8(self.accumulator), True)
            case Operation.MULTIPLY_ACCUMULATE_RESET:
                adc = np.uint8(int(float(input("Enter analog value [0, 1]: ")) * 255))
                self.accumulator = adc * b
                return (np.uint8(self.accumulator), True)
            
            case Operation.CONFIGURE_DAC:
                return (a, False)
            
            case Operation.BREAK:
                self.halt = True
                return (a, False)
            
            case _:
                raise Exception(f"Operation {operation} ({operation.value}) is not implemented")

    def execute_instruction(self):
        # Fetch instruction
        instruction: np.uint16 = self.program[self.pc]

        # Execute instruction
        i_output_register = (instruction >> 12) & 0b111
        store_result: bool = True
        result: np.uint8 = np.uint8(0)
        if (instruction >> 15) == 0:
            # LOAD

            # Decode
            condition = list(Condition)[(instruction >> 9) & 0b111]
            relative = ((instruction >> 8) & 0b1) != 0
            value = np.uint8(instruction & 0xFF)

            # Execute
            store_result = self.evaluate_condition(condition, self.registers[7])
            result = value
            if relative:
                result += self.registers[i_output_register]

        else:
            # OPERATION

            # Decode instruction
            i_input_b_register = (instruction >> 9) & 0b111
            i_input_a_register = (instruction >> 6) & 0b111
            operation = list(Operation)[(instruction & 0b111111)]

            # Execute
            a = self.registers[i_input_a_register]
            b = self.registers[i_input_b_register]
            result, store_result = self.evaluate_operation(operation, a, b)

        # Increment PC
        self.pc += 1

        # Store result
        if store_result:
            self.registers[i_output_register] = result

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
    argument_parser.add_argument("filename")
    argument_parser.add_argument("-t", "--time")
    arguments = argument_parser.parse_args()
    input_filename: str = arguments.filename
    clock_time: float = float(arguments.time if arguments.time is not None else 0.0)

    # Read input lines
    input_filepath = pathlib.Path(input_filename)
    with open(input_filepath, "r") as input_file:
        src_lines = [line.strip() for line in input_file.readlines()]

    # Assemble the program
    program = mcasm8.assemble(src_lines)
    print(f"Assembled {len(program.instructions)} instructions")

    emulator = Emulator()
    emulator.load_program(program.binary)
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
