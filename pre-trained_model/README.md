## How to use

- clone this repository
- cd into pre-trained_model
- `pip install -r requiremens.txt`
- if you would like to run with sample data from paper
  - `python how2use.py`
- if you would like to use different assembly
  - dump assembly from something
  - use the following line to strip offsets and opcodes
    `ndisasm <file> | sed 's/ \+/ /g' | cut -d' ' -f3,4,5,6 > input.asm`
  - `python how2use.py`
