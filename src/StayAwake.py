import ctypes
import sys

# Adapted from http://nullege.com/codes/show/src@c@u@Cura-HEAD@Cura@gui@printWindow.py/23/ctypes.windll.kernel32.SetThreadExecutionState
if sys.platform.startswith('win'):
    def preventComputerFromSleeping(prevent):
        """
        Function used to prevent the computer from going into sleep mode.
        :param prevent: True = Prevent the system from going to sleep from this point on.
        :param prevent: False = No longer prevent the system from going to sleep.
        """
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        #SetThreadExecutionState returns 0 when failed, which is ignored. The function should be supported from windows XP and up.
        if prevent:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
  
else:
    def preventComputerFromSleeping(prevent):
        #No preventComputerFromSleeping for MacOS and Linux yet.
        pass
