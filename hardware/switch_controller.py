import RPi.GPIO as GPIO
import time

class RFSwitchController:
    """
    Controls Mini-Circuits ZFSWA-2-46 switches via GPIO relays.
    Switch control: RF1=+5V/RF2=GND = Port 1, RF1=GND/RF2=+5V = Port 2
    """
    
    def __init__(self, tx_switch_pins, rx_switch_pins):
        """
        Args:
            tx_switch_pins: [relay1_pin, relay2_pin] for TX switch (Antennas 1&2)
            rx_switch_pins: [relay1_pin, relay2_pin] for RX switch (Antennas 3&4)
        """
        self.tx_pins = tx_switch_pins  # e.g., [17, 27]
        self.rx_pins = rx_switch_pins  # e.g., [22, 23]
        
        GPIO.setmode(GPIO.BCM)
        for pin in self.tx_pins + self.rx_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)  # Start with relays OFF
        
        print(f"Switch controller initialized")
        print(f"TX switch on pins: {self.tx_pins}")
        print(f"RX switch on pins: {self.rx_pins}")
    
    def _set_single_switch(self, pins, port):
        """
        Set a single switch to specific port (1 or 2).
        
        Relay wiring for one switch:
        - Relay1: Connects RF1 to +5V when HIGH, to GND when LOW
        - Relay2: Connects RF2 to +5V when HIGH, to GND when LOW
        
        Port 1: RF1=+5V, RF2=GND -> Relay1=HIGH, Relay2=LOW
        Port 2: RF1=GND, RF2=+5V -> Relay1=LOW, Relay2=HIGH
        """
        if port == 1:
            GPIO.output(pins[0], GPIO.HIGH)  # RF1 = +5V
            GPIO.output(pins[1], GPIO.LOW)   # RF2 = GND
        elif port == 2:
            GPIO.output(pins[0], GPIO.LOW)   # RF1 = GND
            GPIO.output(pins[1], GPIO.HIGH)  # RF2 = +5V
        else:
            raise ValueError("Port must be 1 or 2")
        
        time.sleep(0.01)  # 10ms for switch settling
    
    def select_antenna_pair(self, tx_antenna, rx_antenna):
        """
        Select specific TX and RX antennas.
        
        Args:
            tx_antenna: 1 or 2 (TX switch position)
            rx_antenna: 3 or 4 (RX switch position - mapped to 1 or 2)
        """
        # Map antenna numbers to switch ports
        tx_port = 1 if tx_antenna == 1 else 2
        rx_port = 1 if rx_antenna == 3 else 2  # Antenna 3=Port1, Antenna4=Port2
        
        # Set switches
        self._set_single_switch(self.tx_pins, tx_port)
        self._set_single_switch(self.rx_pins, rx_port)
        
        # Log the state
        print(f"  Selected: TX Antenna {tx_antenna} (Port {tx_port}), "
              f"RX Antenna {rx_antenna} (Port {rx_port})")
        
        # Extra settling time for first measurement
        if hasattr(self, 'first_measurement'):
            time.sleep(0.02)
        else:
            self.first_measurement = True
            time.sleep(0.05)
    
    def get_antenna_pairs(self):
        """Return all possible antenna pair combinations."""
        return [(1, 3), (1, 4), (2, 3), (2, 4)]
    
    def cleanup(self):
        """Clean up GPIO."""
        for pin in self.tx_pins + self.rx_pins:
            GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()
        print("GPIO cleaned up")