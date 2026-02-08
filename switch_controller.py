class RFSwitchController:
    def __init__(self, tx_switch_gpio1, tx_switch_gpio2, rx_switch_gpio1, rx_switch_gpio2):
        """Control both RF switches via GPIO pins that will control relays."""
        self.tx_pins = [tx_switch_gpio1, tx_switch_gpio2]
        self.rx_pins = [rx_switch_gpio1, rx_switch_gpio2]
        self.setup_gpio()
    
    def select_antenna_pair(self, tx_ant, rx_ant):
        """
        Select specific TX and RX antennas.
        tx_ant: 1 or 2 (for switch #1)
        rx_ant: 3 or 4 (for switch #2)
        """
        # Control TX switch (Antennas 1 & 2)
        if tx_ant == 1:
            self._set_switch(self.tx_pins, 'RF1_HIGH')  # RF1=+5V, RF2=GND
        else:  # tx_ant == 2
            self._set_switch(self.tx_pins, 'RF2_HIGH')  # RF1=GND, RF2=+5V
        
        # Control RX switch (Antennas 3 & 4)
        if rx_ant == 3:
            self._set_switch(self.rx_pins, 'RF1_HIGH')
        else:  # rx_ant == 4
            self._set_switch(self.rx_pins, 'RF2_HIGH')
    
    def _set_switch(self, pins, mode):
        """Simulate relay control for RF1/RF2 ports."""
        # This will control actual relays when hardware is connected
        print(f"  Setting pins {pins} to mode: {mode}")
        # GPIO.output(pins[0], HIGH if 'RF1' in mode else LOW)
        # GPIO.output(pins[1], HIGH if 'RF2' in mode else LOW)
        time.sleep(0.01)  # Switch settling time
