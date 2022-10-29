import os
from dynamixel_sdk import *

if os.name == 'nt':
    import msvcrt


    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)


    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class Motor:
    def __init__(self, device_index=0):

        if device_index == 0:
            DEVICENAME = '/dev/ttyUSB0'  # USB used
        else:
            DEVICENAME = '/dev/ttyUSB1'  # USB used

        #  ------------------Initial Parameters---------------------------
        # Address of each parameter. See the eManual for details about these values
        self.ADDR_TORQUE_ENABLE  = 24
        self.ADDR_GOAL_POSITION  = 30
        self.ADDR_MOVING_SPEED   = 32
        self.ADDR_TORQUE_LIMIT   = 35
        self.ADDR_LED_ENABLE     = 25
        self.ADDR_PRESENT_POSITION = 37

        self.BAUDRATE = 1000000  # Default Baudrate of XL-320 is 1Mbps
        self.PROTOCOL_VERSION = 2.0  # Default for XL-320

        # ID for each motor
        self.DXL_ID_1 = 1
        self.DXL_ID_2 = 2
        self.DXL_ID_3 = 3
        self.DXL_ID_4 = 4
        self.motor_list = [1, 2, 3, 4]

        # Configuration values
        self.TORQUE_ENABLE = 1  # Value for enabling the torque
        self.TORQUE_DISABLE = 0  # Value for disabling the torque
        self.DXL_MAX_VELOCITY_VALUE = 100  # Value for limited the speed. Max possible value=2047 meaning max speed
        self.DXL_MAX_TORQUE_VALUE   = 100  # It is the torque value of maximum output. 0 to 1,023 can be used


        self.portHandler = PortHandler(DEVICENAME)
        # Initialize PacketHandler instance
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # open the port
        self.open_usb_port()

        for motor_id in self.motor_list:
            self.torque_limit(motor_id)
            self.torque_enable(motor_id)
            self.speed_limit(motor_id)
        print("-------------------------------------------------------------")

        # ---------------------Initialize GroupSyncWrite instance --------------------
        # Need this in order to move all the motor at the same time
        # Initialize GroupSyncWrite instance ---> GroupSyncWrite(port, ph, start_address, data_length)

        data_length = 2  # data len of goal position and present position
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, data_length)

    def open_usb_port(self):
        print("-------------------------------------------------------------")
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()
        print("-------------------------------------------------------------")

    def torque_limit(self, ID):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, ID,
                                                                       self.ADDR_TORQUE_LIMIT,
                                                                       self.DXL_MAX_TORQUE_VALUE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully limited the TORQUE" % ID)

    def torque_enable(self, ID):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, ID,
                                                                       self.ADDR_TORQUE_ENABLE,
                                                                       self.TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully enable torque" % ID)

    def speed_limit(self, ID):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, ID,
                                                                       self.ADDR_MOVING_SPEED,
                                                                       self.DXL_MAX_VELOCITY_VALUE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully limited the speed" % ID)

    def read_servo_position(self, motor_id):
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler,
                                                                                            motor_id,
                                                                                            self.ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        return dxl_present_position

    def get_angles(self):
        # Arm 1
        # position in steps values
        pos_m1_arm_1 = self.read_servo_position(self.DXL_ID_1)
        pos_m2_arm_1 = self.read_servo_position(self.DXL_ID_2)

        # Arm 2
        # position in steps values
        pos_m3_arm_2 = self.read_servo_position(self.DXL_ID_3)
        pos_m4_arm_2 = self.read_servo_position(self.DXL_ID_4)

        # Values in degrees
        tetha_1_arm_1 = pos_m1_arm_1 * 0.29326
        tetha_2_arm_1 = pos_m2_arm_1 * 0.29326

        tetha_1_arm_2 = pos_m3_arm_2 * 0.29326
        tetha_2_arm_2 = pos_m4_arm_2 * 0.29326

        # IMPORTANT, need these values in order to match the equations
        tetha_1_arm_1 = tetha_1_arm_1 - 60
        tetha_1_arm_2 = tetha_1_arm_2 - 60

        tetha_2_arm_1 = 150 - tetha_2_arm_1
        tetha_2_arm_2 = 150 - tetha_2_arm_2

        return tetha_1_arm_1, tetha_2_arm_1, tetha_1_arm_2, tetha_2_arm_2

    def motor_terminate(self):
        # Disable Torque for each motor
        for motor_id in self.motor_list:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id,
                                                                           self.ADDR_TORQUE_ENABLE,
                                                                           self.TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d has been successfully disable torque" % motor_id)

    def motor_led(self):
        # -------> Turn on/off  motor's led <--------------
        # Off = 0
        # Blue= 4
        Color = [0, 4]
        i = 0
        for motor_id in self.motor_list:
            for flash in range(10):
                time.sleep(0.1)
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler,
                                                                               motor_id,
                                                                               self.ADDR_LED_ENABLE,
                                                                               Color[i])
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))

                # Change color position index
                if i == 0:
                    i = 1
                else:
                    i = 0

            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler,
                                                                           motor_id,
                                                                           self.ADDR_LED_ENABLE,
                                                                           0)


    def move_motor_step(self, id_1_dxl_goal_position, id_2_dxl_goal_position, id_3_dxl_goal_position, id_4_dxl_goal_position):

        # -------> This function move the motors i.e. take the actions  <--------------------------------
        param_goal_position_1 = [DXL_LOBYTE(id_1_dxl_goal_position), DXL_HIBYTE(id_1_dxl_goal_position)]
        param_goal_position_2 = [DXL_LOBYTE(id_2_dxl_goal_position), DXL_HIBYTE(id_2_dxl_goal_position)]
        param_goal_position_3 = [DXL_LOBYTE(id_3_dxl_goal_position), DXL_HIBYTE(id_3_dxl_goal_position)]
        param_goal_position_4 = [DXL_LOBYTE(id_4_dxl_goal_position), DXL_HIBYTE(id_4_dxl_goal_position)]

        # --- Add the goal position value to the GroupSync, motor ID1 ----
        dxl_addparam_result = self.groupSyncWrite.addParam(self.DXL_ID_1, param_goal_position_1)
        if dxl_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % self.DXL_ID_1)
            quit()

        # --- Add the goal position value to the GroupSync, motor ID2 ----
        dxl_addparam_result = self.groupSyncWrite.addParam(self.DXL_ID_2, param_goal_position_2)
        if dxl_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % self.DXL_ID_2)
            quit()

        # --- Add the goal position value to the GroupSync, motor ID3 ----
        dxl_addparam_result = self.groupSyncWrite.addParam(self.DXL_ID_3, param_goal_position_3)
        if dxl_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % self.DXL_ID_3)
            quit()

        # --- Add the goal position value to the GroupSync, motor ID4 ----
        dxl_addparam_result = self.groupSyncWrite.addParam(self.DXL_ID_4, param_goal_position_4)
        if dxl_addparam_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % self.DXL_ID_4)
            quit()

        # ---- Transmits packet (goal positions) to the motors
        dxl_comm_result = self.groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        self.groupSyncWrite.clearParam()

        start_time = time.time()
        timer = 0

        # read the current position and check if the motor reaches the desired position
        while True:
            present_step_pos_serv_1 = self.read_servo_position(self.DXL_ID_1)
            present_step_pos_serv_2 = self.read_servo_position(self.DXL_ID_2)
            present_step_pos_serv_3 = self.read_servo_position(self.DXL_ID_3)
            present_step_pos_serv_4 = self.read_servo_position(self.DXL_ID_4)

            if (    (abs(id_1_dxl_goal_position - present_step_pos_serv_1) < 5) and
                    (abs(id_2_dxl_goal_position - present_step_pos_serv_2) < 5) and
                    (abs(id_3_dxl_goal_position - present_step_pos_serv_3) < 5) and
                    (abs(id_4_dxl_goal_position - present_step_pos_serv_4) < 5)):
                break

            end_time = time.time()
            timer = end_time - start_time

            if timer >= 1.0:
                #print("time over, couldn't reach to the point. Moving to next action")
                break
