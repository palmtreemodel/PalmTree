# 🌴PalmTree

This is the official repository for 🌴PalmTree, which is a pre-train Language model for assembly. We will actively update it.
Currently supported platforms: x86

You can find pre-trained PalmTree model [here](https://drive.google.com/file/d/1yC3M-kVTFWql6hCgM_QCbKtc1PbdVdvp/view?usp=sharing)

please consider citing our paper
Xuezixiang Li, Yu Qu, and Heng Yin, "PalmTree: Learning an Assembly Language Model for Instruction Embedding", [CCS'2021]

## Requirements:
- cuda >= 10.1
- pytorch >= 1.3.1
- binary ninja (optional, for dataset generation)

## Updates

### extrinsic evaluations
We released code for extrinsic evaluations. But it is not stable yet. Please checkout to the new preview branch.

Extrinsic evluations including:
- Gemini
- EKALVYA
- DeepVSA

### intrinsic evaluations
Intrinsic evlautions including:
- Opcode Outlier detection
- Operand Outlier detection
- Basicblock matching

### Dataset generator

dataflow_gen.py: generate dataflow dataset for PalmTree model via Binary Ninja and Binary ninja mid-level IR. 
control_flow_gen.py: generate control flow dataset for PalmTree model using Binary Ninja API.

### Dataset format
TXT files. Data-flow and control-flow graph will be sampled by random walk algorithm. And then splitted into instruction pairs.
For a given instruction sequence
```
push rbp
mov rbp, rsp
sub rsp, 0x20
```
Two lines of instruction pairs will be generated:
```
push rbp    mov rbp rsp
mov rbp rsp sub rsp 0x20
```
In detail:
```
push\<SPACE\>rbp<\t>mov\<SPACE\>rbp\<SPACE\>rsp<\n>
```
## TODO:

- Support more binary tools (Ghidra, IDA pro, etc.)
- Support Docker
