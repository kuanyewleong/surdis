///////////////////////////////////////////////////////////////////////////////////
// This code runs DUO Camera MLX on Odroid XU3
// with physical buttons connected to GPIOs to turn the camera on/off 
// and save the IMU data into CSV format, and the video as JPG sequences.
// DUO SDK from Code Laboratories, Inc. is needed.
// Written and updated 2016 by Kuan Yew, Leong. 
// Latest update: 18 Oct 2016
///////////////////////////////////////////////////////////////////////////////////
#include "duo3d.h"

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>

// includes for OpenCV
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

// includes for GPIO                                                                                                                        
#include <sys/mman.h>                                                           
#include <stdint.h>

// include for reading file (iostream objects)
#include <iomanip>

using namespace std;

// define camera frame size and frame rate per second
#define WIDTH	752
#define HEIGHT	480
#define FPS	30

void tiny_delay(void); // needed for debouncing
                                                                               
static volatile uint32_t *gpio;
static volatile uint32_t *gpio2; 


int main(int argc, char* argv[])
{

	// ################ GPIO Configurations ################### //
	int fd ;		                                                              
                                                                                
        if ((fd = open ("/dev/mem", O_RDWR | O_SYNC) ) < 0) {                   
                printf("Unable to open /dev/mem\n");                            
                return -1;                                                      
        }                                                                       
                                                                                
        gpio = (uint32_t *)mmap(0, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd,   
				0x13400000);   // have to type cast the mmap as uint32_t 
        if (gpio < 0){                                                          
                printf("Mmap failed.\n");                                       
                return -1;                                                      
        }

	gpio2 = (uint32_t *)mmap(0, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd,   
			0x14010000); // 0x14010000 is base address for Pin9 and Pin11

        if (gpio2 < 0){                                                          
                printf("Mmap failed.\n");                                       
                return -1;                                                      
        }     
	                                                                 

        // -------------------- Registers Configuration -------------------- //
                                                                        
    // Print GPX1 configuration register.                                   
    printf("GPX1CON register : 0x%08x\n",                                   
                                *(unsigned int *)(gpio + (0x0c20 >> 2)));   

	// Print GPX2 configuration register.                                   
    printf("GPX2CON register : 0x%08x\n",         
                                *(unsigned int *)(gpio + (0x0c40 >> 2)));    
                                                                            
	// Set direction of GPX1.2 (pin15) configuration register as out.               
	*(gpio + (0x0c20 >> 2)) |= (0x1 << 8);                                  
	printf("GPX1CON register : 0x%08x\n",                                   
        		        *(unsigned int *)(gpio + (0x0c20 >> 2)));     

	// Set direction of GPX1.6 (pin17) configuration register as out.               
	*(gpio + (0x0c20 >> 2)) |= (0x1 << 24);                                  
	printf("GPX1CON register : 0x%08x\n",                                   
        		        *(unsigned int *)(gpio + (0x0c20 >> 2))); 

	// Set direction of GPX2.7 (pin22) configuration register as in.             
	*(gpio + (0x0c40 >> 2)) |= (0x0 << 28);                                  
	printf("GPX2CON register : 0x%08x\n",                                   
        		        *(unsigned int *)(gpio + (0x0c40 >> 2)));

	// Set direction of GPX2.1 (pin24) configuration register as out.             
	*(gpio + (0x0c40 >> 2)) |= (0x1 << 4);                                  
	printf("GPX2CON register : 0x%08x\n",                                   
        		        *(unsigned int *)(gpio + (0x0c40 >> 2)));

	// Set direction of GPX3.1 (pin27) configuration register as out.
	*(gpio + (0x0c60 >> 2)) |= (0x1 << 4);                                  
	printf("GPX3CON register : 0x%08x\n",                                   
        		        *(unsigned int *)(gpio + (0x0c60 >> 2)));

	// Set direction of GPA2.5 (pin11) configuration register as in.             
	*(gpio2 + (0x0040 >> 2)) |= (0x0 << 20);                                  
	printf("GPA2CON register : 0x%08x\n",                                   
        		        *(unsigned int *)(gpio2 + (0x0040 >> 2)));

	// Set direction of GPA2.6 (pin9) configuration register as out.             
	*(gpio2 + (0x0040 >> 2)) |= (0x1 << 24);                                  
	printf("GPA2CON register : 0x%08x\n",                                   
        		        *(unsigned int *)(gpio2 + (0x0040 >> 2))); 



	// ----------------- Registers Setting (HIGH / LOW) --------------- //

	// Pull-Up Resister between Pin17 and 22
	// Set GPX1.6 (pin17) HIGH 
	// (input Pin22 is alwasys HIGH due to input from Pin17, until a button is pressed)                                                          
        *(gpio + (0x0c24 >> 2)) |= (1 << 6);                                   
        printf("GPX1DAT register : 0x%08x\n",                                   
                                *(unsigned int *)(gpio + (0x0c24 >> 2)));

	// Pull-Up Resister between Pin9 and 11
	// Set GPA2.6 (pin9) HIGH 
	// (input Pin11 is alwasys HIGH due to input from Pin9, until a button is pressed)                                                          
        *(gpio2 + (0x0044 >> 2)) |= (1 << 6);                                   
        printf("GPA2DAT register : 0x%08x\n",                                   
                                *(unsigned int *)(gpio2 + (0x0044 >> 2)));

	// GPX1.2 (pin15) High, indicating application is ready                                                          
  	*(gpio + (0x0c24 >> 2)) |= (1 << 2);


	// ################### End of GPIO Configurations ################### //


	
	int flag=0, pin22Previous, pin22Current; // variables for handling debouncing

	pin22Previous = *(gpio + (0x0c44 >> 2)) & (1 << 7); // Initial Sample for Debouncing


	
	// first prepare a counter by reading a number from number.txt
	// then use the counter to manage the folder names for each session, also for file names of IMU data
	const char* filename = "/home/odroid/bin/number.txt";
	std::ifstream inFile(filename);		
	int get_counter = 0;
	inFile >> get_counter;
	int main_counter = get_counter; 


	// Turning LED on Pin15 & Pin27 on/off when a button placed btw Pin22 and GRD is pressed
	// causing Pin22 to be LOW
	// Also turn the camera capture ON/OFF when the button is pressed
	int pin11Previous, pin11Current;

	pin11Previous = *(gpio2 + (0x0044 >> 2)) & (1 << 5);

	int termination = 0; // a flag to terminate the loop

	while(termination == 0)	// shutdown Odroid yes/no?
	{
		pin22Current = *(gpio + (0x0c44 >> 2)) & (1 << 7); // New Sample for Debouncing

		if (pin22Current != pin22Previous)		
		{		
			tiny_delay(); // momentary delay

			// Read again current state of Pin22
			pin22Current = *(gpio + (0x0c44 >> 2)) & (1 << 7);
			

			if (pin22Current != pin22Previous)
			{
				// State of Pin22 has indeed changed
				if(flag)
				{
					// GPX1.2 (pin15) High                                                          
        				*(gpio + (0x0c24 >> 2)) |= (1 << 2);                                    
        				printf("GPX1DAT register : 0x%08x\n",                                   
                                			*(unsigned int *)(gpio + (0x0c24 >> 2)));    
						printf("GPX1.2 (Pin 15) HIGH\n");

					// Set GPX3.1 (pin27) LOW                                                          
        				*(gpio + (0x0c64 >> 2)) &= ~(1 << 1);                                    
        				printf("GPX3DAT register : 0x%08x\n",                                   
                                			*(unsigned int *)(gpio + (0x0c64 >> 2)));

					flag = 0;
				}
				else
				{
					// GPX1.2 (pin15) Low                                                           
        				*(gpio + (0x0c24 >> 2)) &= ~(1 << 2);                                   
        				printf("GPX1DAT register : 0x%08x\n",                                   
                               		 		*(unsigned int *)(gpio + (0x0c24 >> 2)));
						printf("GPX1.2 (Pin 15) LOW\n");

					// Set GPX3.1 (pin27) HIGH                                                          
        				*(gpio + (0x0c64 >> 2)) |= (1 << 1);                                    
        				printf("GPX3DAT register : 0x%08x\n",                                   
                                			*(unsigned int *)(gpio + (0x0c64 >> 2)));
					
					// ######## Start-up DUO Camera ######## //
					int duoFrameNum = 0;
						
					printf("DUOLib Version:       v%s\n", GetLibVersion());

					// Open DUO camera 
					if(!OpenDUOCamera(WIDTH, HEIGHT, FPS))
					{
						printf("Could not open DUO camera\n");
						return 0;
					}						
	
					// Create OpenCV windows
					cvNamedWindow("Left");
					cvNamedWindow("Right");

					// Create image headers for left & right frames
					IplImage *left = cvCreateImageHeader(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1);
					IplImage *right = cvCreateImageHeader(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1);


					// ------ Configuring the DUO Camera ----- //
					// Set exposure and LED brightness
					SetGain(35);
					SetExposure(25);
					//SetDUOAutoExposure(true);
					SetLed(25);

					// Set Vertical Flip to "true"
					SetDUOVFlip(_duo, true);

					// Set Horizotal Flip to "true"
					SetDUOHFlip(_duo, true);

					// Set Undistortion to "true"
					SetDUOUndistort(_duo, true);
					// ------------------------------------- //

					
					// ---- Open and prepare a CSV file for recording of IMU data ---- //
					ofstream IMUfile;
					stringstream ss_IMU;  // use stringstream ss to concatenate the filename later
					string IMU_1 = "/media/odroid/MYLINUXLIVE/IMU/IMUData-";
					string CSVtype = ".csv";
					ss_IMU<<IMU_1<<(main_counter)<<CSVtype;
					//convert stringstream to char as needed by file.open()
					char* filename_IMU = new char[ss_IMU.str().length()+1]; 
					ss_IMU >> filename_IMU;	
					IMUfile.open (filename_IMU);
					// ------------------------------------------------------------- //	
	

					// Run camera capture, loop until button is pressed (that changes the input of Pin 22)
					pin22Previous = *(gpio + (0x0c44 >> 2)) & (1 << 7);

					int stop_camera = 0; // a flag to stop the camera from looping
										
					while(stop_camera == 0)	// Camera loop
					{													
						cvWaitKey(10); // a small interval of 10 milliseconds is needed 
							       // to display the image with cvShowImage()

						// Capture DUO frame
						PDUOFrame pFrameData = GetDUOFrame();
						if(pFrameData == NULL) continue;

						// Set the image data
						left->imageData = (char*)pFrameData->leftData;
						right->imageData = (char*)pFrameData->rightData;

						
						// Print text data (comment them when running in Odroid)
						printf("DUO Frame #%d\n", duoFrameNum++);		
						printf("  Timestamp:          %10.1f ms\n", pFrameData->timeStamp/10.0f);
						printf("  Frame Size:         %dx%d\n", pFrameData->width, pFrameData->height);
						printf("  Left Frame Buffer:  %p\n", pFrameData->leftData);
						printf("  Right Frame Buffer: %p\n", pFrameData->rightData);
						printf("------------------------------------------------------\n");

						// Blink LED on Pin 24 while writing data to flashdrive (HIGH)
						*(gpio + (0x0c44 >> 2)) |= (1 << 1);
		
						// Print Frame and Timestamp on each cycle of the sampling (into the CSV file)
						IMUfile << "Frame:" << "," << duoFrameNum << "," << "Timestamp:" << "," << pFrameData->timeStamp/10.0f << "\n";
		
						if(pFrameData->IMUPresent)
						{

							for(int i = 0; i < pFrameData->IMUSamples; i++)
							{	// comment the following printf lines when running in Odroid
								printf(" Sample #%d\n", i+1);
								printf("  Accelerometer: [%8.5f, %8.5f, %8.5f]\n", pFrameData->IMUData[i].accelData[0], 
														   pFrameData->IMUData[i].accelData[1], 
														   pFrameData->IMUData[i].accelData[2]);
								printf("  Gyro:          [%8.5f, %8.5f, %8.5f]\n", pFrameData->IMUData[i].gyroData[0], 
														   pFrameData->IMUData[i].gyroData[1], 
														   pFrameData->IMUData[i].gyroData[2]);
								printf("  Temperature:   %8.6f C\n", pFrameData->IMUData[i].tempData); 

								IMUfile << i+1 << "," 
									<< pFrameData->IMUData[i].accelData[0] << "," 
									<< pFrameData->IMUData[i].accelData[1] << "," 
									<< pFrameData->IMUData[i].accelData[2] << ","
									<< pFrameData->IMUData[i].gyroData[0] << "," 
									<< pFrameData->IMUData[i].gyroData[1] << "," 
									<< pFrameData->IMUData[i].gyroData[2] << ","
									<< pFrameData->IMUData[i].tempData << "\n";				
							}


						}

		
						// Process images here (optional)

						// Display images, can comment the following 2 lines out if going outdoor for data collection
						cvShowImage("Left", left);
						cvShowImage("Right", right);
						

						// ----- writing image sequence to files ----- 		
				
						// for left images... 
						stringstream ss_left;  // use stringstream ss to concatenate the filename later
						string name_left = "/media/odroid/MYLINUXLIVE/Image/left-image-folder/left_";
						string name_left2 = "/l_img_";
						string type = ".jpg";
						ss_left<<name_left<<(main_counter)<<name_left2<<(duoFrameNum)<<type;
						//convert stringstream to char as needed by cvSaveImage()
						char* filename_left = new char[ss_left.str().length()+1]; 
						ss_left >> filename_left;		
						// Write image sequence to file		
						cvSaveImage(filename_left, left);		

						// for right images... 
						stringstream ss_right;  // use stringstream ss to concatenate the filename later
						string name_right = "/media/odroid/MYLINUXLIVE/Image/right-image-folder/right_";
						string name_right2 = "/r_img_";
						ss_right<<name_right<<(main_counter)<<name_right2<<(duoFrameNum)<<type;
						//convert stringstream to char as needed by cvSaveImage()
						char* filename_right = new char[ss_right.str().length()+1]; 
						ss_right >> filename_right;		
						// Write image sequence to file		
						cvSaveImage(filename_right, right);

						// Blink LED on Pin 24 while writing data to flashdrive (LOW)
						*(gpio + (0x0c44 >> 2)) &= ~(1 << 1);					
						
						pin22Current = *(gpio + (0x0c44 >> 2)) & (1 << 7); // New Sample for Debouncing

						if (pin22Current != pin22Previous)		
						{		
							tiny_delay(); // momentary delay

							// Read again current state of Pin22
							pin22Current = *(gpio + (0x0c44 >> 2)) & (1 << 7);			

							if (pin22Current != pin22Previous)
							{
								stop_camera = 1; // this will stop the while loop
							}
						}
								
		
					} // ----- End of Camera loop -----

					// Closing of the CSV file for IMU data	
					IMUfile.close();					
	
					// Release image headers
					cvReleaseImageHeader(&left);
					cvReleaseImageHeader(&right);
			
					// Close DUO camera
					CloseDUOCamera();
									
				 	flag = 1;

					// Increase the main_counter here
					main_counter++;
					

				}

				// Set current value as previous since it has been processed
				pin22Previous = pin22Current;

				
			}
		}

		pin11Current = *(gpio2 + (0x0044 >> 2)) & (1 << 5); // New Sample for Debouncing

		if (pin11Current != pin11Previous)		
		{		
			tiny_delay(); // momentary delay

			// Read again current state of Pin11
			pin11Current = *(gpio2 + (0x0044 >> 2)) & (1 << 5);			

			if (pin11Current != pin11Previous)
			{
				termination = 1; // this will stop the while loop
			}
		}
	
	} // End of the while loop

	
	// Prepare and write to number.txt the value of main_counter
	// such that the next time when the system reboot or startup
	// the naming of the folder could start from the value from number.txt	
	ofstream OFileObject;  // create object of Ofstream
	OFileObject.open ("/home/odroid/bin/number.txt"); // Open a file or create it if not present
	OFileObject << main_counter; // writing an integer to file
	OFileObject.close(); // close the file

	// Shut down Odroid
	system("/home/odroid/bin/shutOdroidDown.sh");

	return 0;
}


void tiny_delay(void)
{
	int z,c;
	c = 0;
	for (z=0;z<1500;z++) // may set higher value for higher clock speed
	{
		c++;
	}
}

