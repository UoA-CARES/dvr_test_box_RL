
import logging
from cares_lib.dynamixel.Servo import DynamixelServoError
from gripper_configuration import Gripper

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


# Example of how to use Gripper
def main():
    gripper = Gripper(device_name="/dev/ttyUSB1", baudrate=1000000)
    # These calls to setup the gripper will work or alert the operator to any issues to resolve them, if operator can't resolve it will simply crash with an exception
    gripper.enable()
    gripper.home()

    # Will run and alert the operator if it can't be resolved, if operate can't resolve it then it will return to this except block to exit gracefully outside of the gripper class.
    try:
        target_steps = [410, 410, 590, 590]
        gripper.move(target_steps)
    except DynamixelServoError as error:
        # Handle the error gracefully here as required...
        logging.error(error)
        exit()  # kill the program entirely as gripper is unrecoverable for whatever reason


if __name__ == "__main__":
    main()
