from mcpc import *
import numpy as np

class Emulator:
    program = np.zeros(PROGRAM_MEMORY_SIZE, dtype=np.uint16)
    registers = np.zeros(8, dtype=np.uint8)
    memory = np.zeros(16, dtype=np.uint8)

    @property
    def pc(self) -> np.uint8:
        return self.registers[0]
    
    @pc.setter
    def pc(self, value: np.uint8):
        self.registers[0] = value

    def initialize(self):
        self.registers = np.zeros(8, dtype=np.uint8)
        self.memory = np.zeros(16, dtype=np.uint8)

    def load_program(self, program: np.uint16):
        np.copyto(self.program, program)
        self.initialize()

    def evaluate_condition(self, condition: Condition, value: np.uint8) -> bool:
        is_zero = (value == 0)
        is_negative = ((value & 0b0000_0000) != 0)

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

    def evaluate_operation(self, operation: Operation, a: np.uint8, b: np.uint8) -> np.uint8:
        match operation:
            case Operation.CLEAR:
                return np.uint8(0)
            case Operation.FILL:
                return np.uint8(0xFF)
            case Operation.AND:
                return a & b
            case Operation.NAND:
                return ~(a & b)
            case Operation.OR:
                return a | b
            case Operation.NOR:
                return ~(a | b)
            case Operation.XOR:
                return a ^ b
            case Operation.XNOR:
                return ~(a ^ b)
            
            case Operation.INC:
                return a + 1
            case Operation.DEC:
                return a - 1
            case Operation.ADD:
                return a + b
            case Operation.SUB:
                return a - b
            
            case Operation.SHIFT_LEFT:
                return a << 1
            case Operation.SHIFT_RIGHT:
                return a >> 1
            case Operation.ROTATE_LEFT:
                return (a << 1) | ((a >> 7) & 0b1)
            case Operation.ROTATE_RIGHT:
                return (a >> 1) | ((a & 0b1) << 7)
            
            case Operation.CHECK_ALWAYS:
                return self.evaluate_condition(Condition.ALWAYS, a)
            case Operation.CHECK_NEVER:
                return self.evaluate_condition(Condition.NEVER, a)
            case Operation.CHECK_EQUAL:
                return self.evaluate_condition(Condition.EQUAL, a)
            case Operation.CHECK_NOT_EQUAL:
                return self.evaluate_condition(Condition.NOT_EQUAL, a)
            case Operation.CHECK_LESS:
                return self.evaluate_condition(Condition.LESS, a)
            case Operation.CHECK_GREATER_OR_EQUAL:
                return self.evaluate_condition(Condition.GREATER_OR_EQUAL, a)
            case Operation.CHECK_GREATER:
                return self.evaluate_condition(Condition.GREATER, a)
            case Operation.CHECK_LESS_OR_EQUAL:
                return self.evaluate_condition(Condition.LESS_OR_EQUAL, a)
            
            case Operation.MEMORY_LOAD:
                return self.memory[a]
            case Operation.MEMORY_STORE:
                self.memory[a] = b
                return b # Return the value
            
            case Operation.IO_POLL:
                return True # TODO: Actually check if input is available
            case Operation.IO_READ:
                return np.uint8(int(input("[IO] Input a number: ")))
            case Operation.IO_WRITE:
                print(f"[IO] {a}")
                return a # Return the value

    def execute_instruction(self):
        # Fetch instruction
        instruction = self.program[self.pc]

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
                result += value

        else:
            # OPERATION

            # Decode instruction
            i_input_b_register = (instruction >> 9) & 0b111
            i_input_a_register = (instruction >> 6) & 0b111
            operation = list(Operation)[(instruction & 0b111111)]

            # Execute
            a = self.registers[i_input_a_register]
            b = self.registers[i_input_b_register]
            store_result = True
            result = self.evaluate_operation(operation, a, b)

        # Increment PC
        self.pc += 1

        # Store result
        self.registers[i_output_register] = result