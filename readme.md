# Utilities for the MCPC

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
`<out> = <a> - <b>`|Substracts the values of reigster `b` from the register `a` <br>and `b` and stores the result in `out`.
`<out> = inc <a>`|Calculates the value of `a + 1` and stores the result in `out`.
`<out> = dec <a>`|Calculates the value of `a - 1` and stores the result in `out`.
`<out> = shift right <a>`|Shifts the value of `a` one to the right (filling the msb with 0) and stores the value in `out`.
`<out> = shift left <a>`|Shifts the value of `a` one to the left (filling the lsb with 0) and stores the value in `out`.
`<out> = rotate right <a>`|Rotates the value of `a` one to the right (filling the msb with the lsb) and stores the value in `out`.
`<out> = rotate left <a>`|Rotates the value of `a` one to the left (filling the lsb with the msb) and stores the value in `out`.

#### Checks
Instruction|Description
---|---
`<out> = check <r> <condition>`|If the register `r` fulfills the given condition, <br>`1` is stored in `out`. Otherwise `0` is stored in `out`

#### Memory
Instruction|Description
---|---
`<out> = [<r>]`|Uses the value of `r` as memory address and loads a value from memory.
`[<r>] = <v>`|Uses the value of `r` as memory address and stores the value of `v` in memory.

#### Other
Instruction|Description
---|---
`nop`|Does nothing
`clear <r>`|Clears all bits of the register `r`.
`fill <r>`|Sets all bits of the register `r`.