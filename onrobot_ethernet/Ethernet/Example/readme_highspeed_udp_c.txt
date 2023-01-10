Ethernet DAQ - Highspeed UDP C Sample

This is a simple demo showing how to communicate with an Ethernet DAQ
over UDP using standard C functions.

The sample was tested using gcc version 4.8.5 running under Ubuntu 14.04 64-bit
and a Windows 8.1 64-bit using Visual C++ (2010).


------------------------------------------------------------
COMPILING USING GCC
------------------------------------------------------------
Compile with sudo gcc highspeed_udp.c -o highspeed_udp

------------------------------------------------------------
COMPILING USING VISUAL STUDIO
------------------------------------------------------------
Create a new Win32 Console Application project. Remove all
files under Header Files and Source Files, then add highspeed_udp.c
to the Source Files.

------------------------------------------------------------
USING THE SAMPLE
------------------------------------------------------------
Enter the command "highspeed_udp IPADDRESS" at the command line, where the IPADDRESS
token is replaced by the IP address of your Ethernet DAQ, e.g. "192.168.1.1", 
without the quotes.  If the IP address is incorrect, or the Ethernet DAQ does not
respond to the UDP request, the sample will appear to freeze.  In this case,
press Ctrl-C to terminate the program, and check that you have entered the
correct IP address and that the Ethernet DAQ is set up to allow UDP communication.
Also make sure that your computer does not have a software or hardware firewall 
that is preventing you from accessing UDP port 49152 on the Ethernet DAQ.
