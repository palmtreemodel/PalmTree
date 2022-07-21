'''
transfer the instructions to integer
or transfer the integer to instructions
input: int list
output: one integer
'''


def insn2int_inverse(insn_list):
    '''
    transfer the instruction to integer with inverse order
    example:
    [72,137,229] ==>15042888 (72+137*256+229*256*256)
    [243, 15, 16, 13, 205, 0, 0, 0] ==> 880687452147 (243+15*256+16*256*256+13*256*256*256+13*256*256*256*256+205*256*256*256*256*256)
    :param insn_list:
    :return insn_int:
    '''
    insn_int=0
    for idx, value in enumerate(insn_list):
        insn_int = insn_int + value*(256**idx)
    return insn_int


def insn2int(insn_list):
    '''
    transfer the instruction to integer
    example:
    [72,137,229] ==> 4753893 (72*256*256+137*256+229)
    [243, 15, 16, 13, 205, 0, 0, 0] ==>  1.751423513*10^19(243*256^7+15*256^6+16*256^5+13*256^4+205*256^3+0*256^2+0*256^1+0*256^0)
    :param insn_list:
    :return insn_list:
    '''
    insn_len=len(insn_list)-1
    insn_int=0
    for idx, value in enumerate(insn_list):
        insn_int = insn_int + value*(256**(insn_len- idx))
    return insn_int