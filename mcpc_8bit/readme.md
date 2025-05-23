# 8bit MCPC

This repository contains utilities for the 8bit MCPC v0.2, a minecraft redstone computer.

## Assembler

Every assembly instruction maps 1:1 to a machine instruction.

## Comments
If a line contains a `#`, everything to the right of it, including itself is removed before the assembler processes the file. 

### Labels
Labels can be declared in a line by writing `@<name>`. The instruction number of the next instruction is then inserted wherever the lable is used. This can be used for jumps.

### Registers
There are 8 registers `R0`,...,`R7` that can be used by their name. The program counter `R0` additionally has the alias `PC` and the conditional register `R0` has the alias `RC`.

### Values
Integers can be specified in the following formats:
- Decimal: `133`
- Binary: `0b00110010`
- Hexadecimal: `0xE3`

### Conditions
The following conditions are supported. Comparisons are always signed.

Conditition|Description
---|---
`<register> = 0`|If the register equals 0.
`<register> != 0`|If the register does not equal 0.
`<register> > 0`|If the register is greater than 0.
`<register> < 0`|If the register less than 0.
`<register> >= 0`|If the register is greater or equal 0.
`<register> <= 0`|If the register is less or equal 0.

### Instructions

#### Load
Instruction|Description
---|---
`<r> = [value]`|Loads the given value into the given register.
`<r> = [value] if RC {condition}`|Loads the given value into the given register,<br> if the `RC` register fulfills the given condition.
`<r> += [value]`|Adds the given signed value to the given register.
`<r> += [value] if RC {condition}`|Adds the given signed value to the given register,<br> if the `RC` register fulfills the given condition.

#### Jump
Instruction|Description
---|---
`jump [value]`|Jumps to the given address.
`jump [value] if RC {condition}`|Jumps to the given address, if the `RC` register fulfills the given condition.
`skip [n]`|Skips `n` instructions. `n` can be negative to jump back.
`skip [n] if RC {condition}`|Skips `n` instructions, if the `RC` register fulfills the given condition.<br> `n` can be negative to jump back.

#### Operation
Instruction|Description
---|---
`<out> = <a>`|Sets the value of `out` to the value of `a`. 
`<out> = not <a>`|Sets the value of `out` to the value of `not a`. 
`<out> = <a> and <b>`|Calculates the value of `a and b` and stores the result in `out`.
`<out> = <a> nand <b>`|Calculates the value of `not (a and b)` and stores the result in `out`.
`<out> = <a> or <b>`|Calculates the value of `a or b` and stores the result in `out`.
`<out> = <a> nor <b>`|Calculates the value of `not (a or b)` and stores the result in `out`.
`<out> = <a> xor <b>`|Calculates the value of `a xor b` and stores the result in `out`.
`<out> = <a> xnor <b>`|Calculates the value of `not (a xor b)` and stores the result in `out`.
`<out> = <a> + <b>`|Adds the values of the reigsters `a` and `b` and stores the result in `out`.
`<out> = <a> - <b>`|Substracts the value of reigster `b` from the register `a` <br> and stores the result in `out`.
`<out> = <a> * <b>`|Multiplies the values of the reigsters `a` and `b` and stores the result in `out`.
`<out> = <a> / <b>`|Divides the value of reigster `a` through the register `b` <br> and stores the result in `out`.
`<out> = <a> % <b>`|Divides the value of reigster `a` through the register `b` <br> and stores the remainder in `out`.
`<out> = inc <a>`|Calculates the value of `a + 1` and stores the result in `out`.
`<out> = dec <a>`|Calculates the value of `a - 1` and stores the result in `out`.
`<out> = shift right <a>`|Shifts the value of `a` one to the right (filling the msb with 0) and stores the value in `out`.
`<out> = shift left <a>`|Shifts the value of `a` one to the left (filling the lsb with 0) and stores the value in `out`.
`<out> = rotate right <a>`|Rotates the value of `a` one to the right (filling the msb with the lsb) and stores the value in `out`.
`<out> = rotate left <a>`|Rotates the value of `a` one to the left (filling the lsb with the msb) and stores the value in `out`.
`<out> = sqrt <a>`|Calculates the sqrt of `a` and stores it in `out`.
`<out> = random`|Stores a random value in `out`.

#### Bit
Instruction|Description
---|---
`<out> = <r> bit {0-7}`|Checks if the given bit of `r` is set. If thats the case `out` is set to `1` otherwise it is set to `0`.
`<out> bit {0-7} = <r>`|Sets the given bit of `out` to the boolean value of `r`. The other bits of `out` remain unchanged.

#### Checks
Instruction|Description
---|---
`<out> = check <r> <condition>`|If the register `r` fulfills the given condition, <br>`1` is stored in `out`. Otherwise `0` is stored in `out`.

#### Memory
Instruction|Description
---|---
`<out> = [<r>]`|Uses the value of `r` as memory address and loads a value from memory.
`[<r>] = <v>`|Uses the value of `r` as memory address and stores the value of `v` in memory.

#### IO
Instruction|Description
---|---
`<r> = poll`|Checks if input is available. If input is available `1` is written to `r`,<br> else `0` is written to `r`.
`<r> = read`|Reads a value from the input and stores the value in `r`. <br>If no input is available, this blocks until input is available.
`write <r>`|Writes the value of `r` to the output.

### DSP
Instruction|Description
---|---
`<out> = mac <a> <b>`|Uses the MAC unit to multiply `a` and `b` and adds the result to the accumulator.
`<out> = macrs adc <b>`|Sets the accumulator to the product of the current value of the ADC and `b`.
`configure dac <r>`|Configures the DAC with the value of `r`.

#### Other
Instruction|Description
---|---
`nop`|Does nothing.
`break`|Causes the computer to halt at the execution of this command.
`clear <r>`|Clears all bits of the register `r`.
`fill <r>`|Sets all bits of the register `r`.