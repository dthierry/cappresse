#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sample_mods.servo_motor.servo_ import servo_motor_dae


def main():
    mod = servo_motor_dae(2, 2)
    mod.pprint()


if __name__ == '__main__':
    main()
