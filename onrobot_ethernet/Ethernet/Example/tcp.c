#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
typedef SOCKET SOCKET_HANDLE;
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
typedef int SOCKET_HANDLE;
#endif

#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#pragma pack(1)

#define PORT			49151	/* Port the Ethernet DAQ always uses */
#define SAMPLE_COUNT	100		/* 100 incoming samples */

#define READFT				0
#define READCALIBRATIONINFO 1


typedef unsigned int uint32;
typedef int int32;
typedef unsigned short uint16;
typedef short int16;
typedef unsigned char byte;

typedef struct FTResponseStruct {
	uint16 header;
	uint16 status;
	int16 ForceX;
	int16 ForceY;
	int16 ForceZ;
	int16 TorqueX;
	int16 TorqueY;
	int16 TorqueZ;
} FTResponse;


typedef struct CalibrationResponseStruct
{
	uint16 header;
	byte forceUnits;
	byte torqueUnits;
	uint32 countsPerForce;
	uint32 countsPerTorque;
	uint16 scaleFactors[6];
} CalibrationResponse;


typedef struct FTReadCommandStruct
{
	byte command;
	byte reserved[19];
}FTReadCommand;


typedef struct ReadCalibrationCommandStruct {
	byte command;
	byte reserved[19];
} ReadCalibrationCommand;


CalibrationResponse calibrationResponse;  // global variable to hold calibration info datas


/* Sleep ms milliseconds */
static void MySleep(unsigned long ms)
{
#ifdef _WIN32
	Sleep(ms);
#else
	usleep(ms * 1000);
#endif
}


static int Connect(SOCKET_HANDLE * handle, const char * ipAddress, uint16 port)
{
	struct sockaddr_in addr;
	struct hostent *he;
	int err;
#ifdef _WIN32
	WSADATA wsaData;
	WORD wVersionRequested;
	wVersionRequested = MAKEWORD(2, 2);
	WSAStartup(wVersionRequested, &wsaData);
	if (GetLastError() != 0) {
		return -1;
	}
#endif

	*handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#ifdef _WIN32
	if (*handle == INVALID_SOCKET) {
#else
	if (*handle == -1) {
#endif
		fprintf(stderr, "Socket could not be opened.\n");
		return -2;
	}
	he = getnameinfo(ipAddress);
	memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);
	fprintf(stdout, "Connecting to EtherDAQ\r\n");
	fflush(stdout);
	err = connect(*handle, (struct sockaddr *)&addr, sizeof(addr));
	if (err < 0) {
		return -3;
	}
	return 0;
}

static void Close(SOCKET_HANDLE * handle)
{
#ifdef _WIN32
	closesocket(*handle);
	WSACleanup();
#else
	close(*handle);
#endif
}


static void ShowCalibrationInfo(CalibrationResponse * r)
{
	int i;
	if (r == NULL) {
		return;
	}
	fprintf(stdout, "Header: 0x%x\r\nForcfe Units: %u\r\nTorque Units: %u\r\nCounts Per Force: %u\r\nCounts Per Torque: %u\r\n",
		r->header, r->forceUnits, r->torqueUnits, r->countsPerForce, r->countsPerTorque);


	for (i = 0; i < 6; ++i) {
		fprintf(stdout, "scaleFactors[%d]: %u\r\n", i, r->scaleFactors[i]);
	}
	fflush(stdout);
}

static int GetCalibrationInfo(SOCKET_HANDLE *socket)
{
	int i;
	int sendSuccess;
	int readSuccess;
	ReadCalibrationCommand readCommand = { 0 };
	readCommand.command = READCALIBRATIONINFO;
	sendSuccess = send(*socket, (const char *)&readCommand, sizeof(ReadCalibrationCommand), 0);
	if (sendSuccess < 0) {
		return sendSuccess;
	}

	readSuccess = recv(*socket, (char *)&calibrationResponse, sizeof(CalibrationResponse), 0);
	if (readSuccess < 0) {
		return readSuccess;
	}
	calibrationResponse.header = htons(calibrationResponse.header);
	calibrationResponse.countsPerForce = ntohl(calibrationResponse.countsPerForce);
	calibrationResponse.countsPerTorque = ntohl(calibrationResponse.countsPerTorque);
	for (i = 0; i < 6; ++i) {
		calibrationResponse.scaleFactors[i] = htons(calibrationResponse.scaleFactors[i]);
	}
	if (calibrationResponse.header != 0x1234) {
		return -1;
	}
	return 0;
}


static int16 swap_int16(int16 val)
{
	return (val << 8) | ((val >> 8) & 0xFF);
}

static void SwapFTResponseBytes(FTResponse * r)
{
	r->header = htons(r->header);
	r->status = htons(r->status);
	r->ForceX = swap_int16(r->ForceX);
	r->ForceY = swap_int16(r->ForceY);
	r->ForceZ = swap_int16(r->ForceZ);
	r->TorqueX = swap_int16(r->TorqueX);
	r->TorqueY = swap_int16(r->TorqueY);
	r->TorqueZ = swap_int16(r->TorqueZ);
}


static int ReadFT(SOCKET_HANDLE * socket, FTResponse * r)
{
	FTReadCommand readCommand = { 0 };
	int readSuccess;
	int sendSuccess;
	readCommand.command = READFT;
	sendSuccess = send(*socket, (char *)&readCommand, sizeof(FTReadCommand), 0);
	if (sendSuccess < 0) {
		return sendSuccess;
	}
	readSuccess = recv(*socket, (char *)r, sizeof(FTResponse), 0);
	if (readSuccess != sizeof(FTResponse)) {
		return 1;
	}
	SwapFTResponseBytes(r);
	if (r->header != 0x1234) {
		return 2;
	}
	return 0;
}


static void ShowResponse(FTResponse * r)
{

	double Fx = (double)r->ForceX / (double)calibrationResponse.countsPerForce * (double)calibrationResponse.scaleFactors[0];
	double Fy = (double)r->ForceY / (double)calibrationResponse.countsPerForce * (double)calibrationResponse.scaleFactors[1];
	double Fz = (double)r->ForceZ / (double)calibrationResponse.countsPerForce * (double)calibrationResponse.scaleFactors[2];
	double Tx = (double)r->TorqueX / (double)calibrationResponse.countsPerTorque * (double)calibrationResponse.scaleFactors[3];
	double Ty = (double)r->TorqueY / (double)calibrationResponse.countsPerTorque * (double)calibrationResponse.scaleFactors[4];
	double Tz = (double)r->TorqueZ / (double)calibrationResponse.countsPerTorque * (double)calibrationResponse.scaleFactors[5];
	
	if (calibrationResponse.forceUnits == 2 && calibrationResponse.torqueUnits == 3) {
		fprintf(stdout, "Status: %u Fx:%.4f N Fy:%.4f N Fz: %.4f N Tx: %.4f Nm Ty: %.4f Nm Tz: %.4f Nm\r\n", r->status, Fx, Fy, Fz, Tx, Ty, Tz);
	}
	else {
		fprintf(stdout, "Status: %u Fx:%.4f Fy:%.4f Fz: %.4f Tx: %.4f Ty: %.4f Tz: %.4f\r\n", r->status, Fx, Fy, Fz, Tx, Ty, Tz);
	}
	fflush(stdout);
}


int main(int argc, char ** argv)
{
	FTResponse ftResponse;
	unsigned int i;
	int readSuccess;
	SOCKET_HANDLE socketHandle;		/* Handle to UDP socket used to communicate with Ethernet DAQ. */
	if (argc < 2) {
		fprintf(stderr, "Usage: %s IPADDRESS\r\n", argv[0]);
		return -1;
	}
	if (Connect(&socketHandle, argv[1], PORT) != 0) {
		fprintf(stderr, "Could not connect to device...\r\n");
		return -1;
	}
	else {
		fprintf(stdout, "Connected to Ethernet DAQ\r\n");
		fflush(stdout);
	}
	if (GetCalibrationInfo(&socketHandle) != 0) {
		fprintf(stderr, "Could not read calibration info...\r\n");
		return -1;
	}
	ShowCalibrationInfo(&calibrationResponse);
	
	MySleep(2000);  // To allow user to read calibration information sent by Ethernet DAQ

	for (i = 0; i < SAMPLE_COUNT; ++i) {
		readSuccess = ReadFT(&socketHandle, &ftResponse);
		if (readSuccess == 0) {
			ShowResponse(&ftResponse);
		}
		else {
			fprintf(stderr, "Could not read F/T data, error code: %d\r\n", readSuccess);
			break;
		}
	}

	Close(&socketHandle);
	return 0;
}

