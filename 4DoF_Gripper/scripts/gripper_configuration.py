
"""
Gripper Configuration
Modification by David
Original File: https://github.com/UoA-CARES/Gripper-Code/blob/f69b55b0bb525b29f2dd2451889eefe61b0eca19/scripts/Gripper.py
"""

import logging
import time
import backoff
import dynamixel_sdk as dxl
from cares_lib.dynamixel.Servo import Servo, DynamixelServoError


def handle_gripper_error(error):
    logging.error(error)
    logging.info("Please fix the gripper and press enter to try again or x to quit: ")
    value  = input()
    if value == 'x':
        logging.info("Giving up correcting gripper")
        return True
    return False


class GripperError(IOError):
    pass

class Gripper(object):
    def __init__(self,
                 motor_reset="On",
                 num_motors=4,
                 gripper_id=0,
                 device_name="/dev/ttyUSB1",
                 baudrate=1000000,
                 protocol=2.0,
                 torque_limit=210,
                 speed_limit=210):

        self.motor_reset = motor_reset

        # Setup Servor handlers
        self.gripper_id  = gripper_id
        self.device_name = device_name

        self.baudrate = baudrate
        self.protocol = protocol  # NOTE: XL-320 uses protocol 2

        self.port_handler   = dxl.PortHandler(self.device_name)
        self.packet_handler = dxl.PacketHandler(self.protocol)
        self.setup_handlers()

        self.group_sync_write = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, Servo.addresses["goal_position"], 2)
        self.group_sync_read  = dxl.GroupSyncRead(self.port_handler, self.packet_handler, Servo.addresses["current_position"], 2)

        self.group_sync_write_reset = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, Servo.addresses["goal_position"], 2)

        self.servos = {}
        leds = [1, 2, 3, 4]
        min  = [440, 260, 500, 510]
        max  = [500, 510, 580, 760]



        if self.motor_reset == "On":
            try:
                motor_id_reset = 5
                self.reset_motor = Servo(self.port_handler, self.packet_handler, 5, motor_id_reset, torque_limit, speed_limit, 1023, 0)
                self.setup_motor_reset()
            except DynamixelServoError as error:
                raise DynamixelServoError(f"Gripper#{self.gripper_id}: Failed to initialise reset motor") from error
        else:
            try:
                for i in range(0, num_motors):
                    self.servos[i] = Servo(self.port_handler, self.packet_handler, leds[i], i + 1, torque_limit, speed_limit, max[i], min[i])
                self.setup_servos()
            except DynamixelServoError as error:
                raise DynamixelServoError(f"Gripper#{self.gripper_id}: Failed to initialise servos") from error


    def setup_handlers(self):
        if not self.port_handler.openPort():
            error_message = f"Failed to open port {self.device_name}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.info(f"Succeeded to open port {self.device_name}")

        if not self.port_handler.setBaudRate(self.baudrate):
            error_message = f"Failed to change the baudrate to {self.baudrate}"
            logging.error(error_message)
            raise IOError(error_message)

        logging.info(f"Succeeded to change the baudrate to {self.baudrate}")


    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def setup_servos(self):
        try:
            for _, servo in self.servos.items():
                servo.disable_torque()
                servo.limit_torque()
                servo.limit_speed()
                servo.enable_torque()
                servo.turn_on_LED()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to setup servos") from error


    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def setup_motor_reset(self):
        try:
            self.reset_motor.disable_torque()
            self.reset_motor.limit_torque()
            self.reset_motor.limit_speed()
            self.reset_motor.enable_torque()
            self.reset_motor.turn_on_LED()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to setup motor servo") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def current_positions(self):
        try:
            current_positions = []
            for id, servo in self.servos.items():
                servo_position = servo.current_position()
                current_positions.append(servo_position)
            return current_positions
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to read current position") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def current_load(self):
        try:
            current_load = []
            for _, servo in self.servos.items():
                current_load.append(servo.current_load())
            return current_load
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to check load") from error

    def is_moving(self):
        try:
            gripper_moving = False
            for _, servo in self.servos.items():
                gripper_moving |= servo.is_moving()
            return gripper_moving
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to check if moving") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def stop_moving(self):
        try:
            for _, servo in self.servos.items():
                servo.stop_moving()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to stop moving") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def move_servo(self, servo_id, target_step, wait=True, timeout=5):
        if servo_id not in self.servos:
            error_message = f"Dynamixel#{servo_id} is not associated to Gripper#{self.gripper_id}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)
        try:
            servo_pose = self.servos[servo_id].move(target_step, wait=wait, timeout=timeout)
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id} failed while moving Dynamixel#{servo_id}") from error


    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def move(self, steps, wait=True, timeout=3):

        if not self.verify_steps(steps):
            error_message = f"Gripper#{self.gripper_id}: The move command provided is out of bounds: Step {steps}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        for id, servo in self.servos.items():
            servo.target_position = steps[id]
            self.group_sync_write.addParam(id + 1, [dxl.DXL_LOBYTE(steps[id]), dxl.DXL_HIBYTE(steps[id])])

        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result != dxl.COMM_SUCCESS:
            error_message = f"Gripper#{self.gripper_id}: group_sync_write Failed"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        logging.debug(f"Gripper#{self.gripper_id}: group_sync_write Succeeded")
        self.group_sync_write.clearParam()

        try:
            start_time = time.perf_counter()
            while wait and self.is_moving() and time.perf_counter() < start_time + timeout:
                pass
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed while moving") from error

        try:
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to read its position") from error


    def home(self):
        try:
            home_pose = [440, 510, 580, 510]
            current_positions = self.move(home_pose)

            if self.motor_reset == "On":
                servo_reset_home_step = 510
                self.move_motor_reset(servo_reset_home_step)

            return current_positions


        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to Home") from error

    def move_motor_reset(self, servo_reset_home_step):
        self.group_sync_write_reset.addParam(5, [dxl.DXL_LOBYTE(servo_reset_home_step), dxl.DXL_HIBYTE(servo_reset_home_step)])
        dxl_comm_result_reset = self.group_sync_write_reset.txPacket()

        if dxl_comm_result_reset != dxl.COMM_SUCCESS:
            error_message = f"Gripper#{self.gripper_id}: group_sync_write Failed"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        logging.info(f"Gripper#{self.gripper_id}: group_sync_write RESET Succeeded")
        self.group_sync_write_reset.clearParam()
        try:
            start_time = time.perf_counter()
            while True and self.is_moving_motor_reset() and time.perf_counter() < start_time + 3:
                pass
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed while moving") from error
        self.reset_motor.disable_torque()

    def is_moving_motor_reset(self):
        gripper_moving = False
        gripper_moving |= self.reset_motor.is_moving()
        return gripper_moving


    def verify_steps(self, steps):
        # check all actions are within min max of each servo
        for id, servo in self.servos.items():
            if not servo.verify_step(steps[id]):
                logging.warn(f"Gripper#{self.gripper_id}: step for servo {id + 1} is out of bounds")
                return False
        return True

    def action_to_steps(self, action):
        steps = action
        max_action = 1
        min_action = -1
        for i in range(0, len(steps)):
            max = self.servos[i].max
            min = self.servos[i].min
            #steps[i] = steps[i] * (max - min) + min
            steps[i] = int((steps[i] - min_action) * (max - min) / (max_action - min_action)  + min)
        return steps

    def close(self):
        # disable torque
        for _, servo in self.servos.items():
            servo.disable_torque()
        # close port
        self.port_handler.closePort()








