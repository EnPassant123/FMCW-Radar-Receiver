#include <arpa/inet.h> // inet_addr()

#include <netdb.h>

#include <stdio.h>

#include <stdlib.h>

#include <stdint.h>

#include <stdbool.h>

#include <arpa/inet.h> // inet_addr()

#include <netdb.h>

#include <stdio.h>

#include <stdlib.h>

#include <stdint.h>

#include <stdbool.h>

#include <string.h>

#include <strings.h> // bzero()

#include <sys/socket.h>

#include <fftw3.h>

#include <pthread.h>

#include <math.h>

#include <omp.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <unistd.h> // read(), write(), close()

const double SWEEPTIME = 0.001;
const double BANDWIDTH =  100000000;
const double SLOPE = BANDWIDTH / SWEEPTIME;
const double C = 299792458;

#define SAMPLING_RATE 1000000

const double WAVELENGTH = 0.1224;


#define YMINOR 1
#define YMAJOR 5

#define XMINOR 10
#define XMAJOR 50

#define PORT 42069
#define PSIZE 4096

#define FFTSIZE 1024
#define TIMESIZE 512

#define PXTIME 2
#define PXFFT 4

#define XZOOM 4
#define YZOOM 4

#define ZEROSAMPLES 10

#define SAMPLELOWERBOUND 900

#define DISTOFFSET 4

const int16_t MASK = 0x4000;
const int16_t NMASK = ~MASK;
#define SA struct sockaddr

const int WIDTH = FFTSIZE * PXFFT / 2;
const int HEIGHT = TIMESIZE * PXTIME; 

double fft_in[FFTSIZE][TIMESIZE][2];
double fft_out[FFTSIZE][TIMESIZE][2];

double hann[FFTSIZE][TIMESIZE];

pthread_mutex_t fft_start_lock;
pthread_mutex_t fft_stop_lock;

fftw_complex * in;
fftw_complex * out;
fftw_plan p;

volatile int fftctr = 0;
volatile int prevfftctr = 0;

SDL_Event event;
SDL_Renderer * renderer;
SDL_Window * window;


int distanceToX(double distance){
	distance -= DISTOFFSET;
	double frequency = SLOPE * 2 * distance / C;	
	double fftbin = frequency  / ( SAMPLING_RATE / FFTSIZE);
	fftbin *= PXFFT * XZOOM;
	return (int)(fftbin);
}

int velocityToY(double velocity){
	double mid = HEIGHT / 2.0;
	double velRes = WAVELENGTH / ( 2 * SWEEPTIME * TIMESIZE );
        double bin = velocity / velRes;
	bin *= PXTIME * YZOOM;
	return (int)(bin + mid);	
}

struct RGB {
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

struct HSV {
    double H;
    double S;
    double V;
};

struct RGB HSVToRGB(struct HSV hsv) {
    double r = 0, g = 0, b = 0;

    if (hsv.S == 0) {
        r = hsv.V;
        g = hsv.V;
        b = hsv.V;
    } else {
        int i;
        double f, p, q, t;

        if (hsv.H == 360)
            hsv.H = 0;
        else
            hsv.H = hsv.H / 60;

        i = (int) trunc(hsv.H);
        f = hsv.H - i;

        p = hsv.V * (1.0 - hsv.S);
        q = hsv.V * (1.0 - (hsv.S * f));
        t = hsv.V * (1.0 - (hsv.S * (1.0 - f)));

        switch (i) {
        case 0:
            r = hsv.V;
            g = t;
            b = p;
            break;

        case 1:
            r = q;
            g = hsv.V;
            b = p;
            break;

        case 2:
            r = p;
            g = hsv.V;
            b = t;
            break;

        case 3:
            r = p;
            g = q;
            b = hsv.V;
            break;

        case 4:
            r = t;
            g = p;
            b = hsv.V;
            break;

        default:
            r = hsv.V;
            g = p;
            b = q;
            break;
        }

    }

    struct RGB rgb;
    rgb.R = r * 255;
    rgb.G = g * 255;
    rgb.B = b * 255;

    return rgb;
}

void convertToMagSq(double x[FFTSIZE][TIMESIZE][2]) {
    #pragma omp parallel num_threads(4) 
    {
        int start = omp_get_thread_num() / 4 * FFTSIZE;
        int stop = (omp_get_thread_num() + 1) / 4 * FFTSIZE;

        for (int i = start; i < stop; i++) {
            for (int j = 0; j < TIMESIZE; j++) {
                x[i][j][0] = log(x[i][j][0] * x[i][j][0] + x[i][j][1] * x[i][j][1]);
            }
        }

    }
    printf("Converted to magnitude\n");
}

void writeToFile(double x[FFTSIZE][TIMESIZE][2]) {
    char filename[] = "/home/eggert/eggert.csv";
    FILE * fp;
    fp = fopen(filename, "w");

    for (int i = 0; i < FFTSIZE; i++) {
        for (int j = 0; j < TIMESIZE; j++) {
            fprintf(fp, "%f, ", x[i][j][0]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Written to file\n");
}

double map(double x, double in_min, double in_max, double out_min, double out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void drawToScreen(double x[FFTSIZE][TIMESIZE][2]) {
    double avg = 0;
    double max = -99999;
    for (int i = 0; i < FFTSIZE; i++) {
        for (int j = 0; j < TIMESIZE; j++) {
            double value = x[i][j][0];
            avg +=  value;
            max = max < value ? value : max;
        }
    }
    avg /= (FFTSIZE * TIMESIZE);
    for (int i = 0; i < FFTSIZE / 2 / XZOOM; i++) {
        for (int m = - TIMESIZE / 2 / YZOOM; m < TIMESIZE / 2 / YZOOM ; m++) {
	    int j = m >= 0 ? m : TIMESIZE + m;
            double hue = map(x[i][j][0], avg, max, 315, -45);
	    double value = 1;
	    double saturation = 1;
	    hue = hue < -45? -45:hue;
	    hue = hue > 315 ? 315:hue;
	    if (hue < 0){
		saturation = map(hue, -45, 0, 0.001, 1);
		hue = 0;
	    }
	    if (hue > 270){
		    value = map(hue, 270, 315, 1, 0.001);
		    hue = 270;
	    }
	    
            struct HSV hsvcolor = {hue, saturation, value};
            struct RGB rgbcolor = HSVToRGB(hsvcolor);
            SDL_SetRenderDrawColor(renderer, rgbcolor.R, rgbcolor.G, rgbcolor.B, 255);
	    j = m >=0 ? m :  TIMESIZE  / YZOOM + m ;
            int height = j <= TIMESIZE / (2 * YZOOM) ? TIMESIZE /(2 * YZOOM) - j : TIMESIZE * 3 / (2 * YZOOM) - j;
            height *= PXTIME * YZOOM;
	    int width = i * PXFFT * XZOOM;
            for (int k = 0; k < PXTIME * YZOOM; k++) {
		    for (int l = 0 ; l < PXFFT * XZOOM; l++){
                        SDL_RenderDrawPoint(renderer, width+l, height + k);
		    }
            }
        }
    }
    SDL_RenderPresent(renderer);
}

void drawHorizLine(int y){
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 128);
	SDL_RenderDrawLine(renderer, 0, y, WIDTH, y); 
}

void drawVertLine(int x){
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 128);
	SDL_RenderDrawLine(renderer, x, 0, x, HEIGHT);
}

void drawText(int x, int y, char* text, bool centered){
	const int size = 30;
	TTF_Font* font = TTF_OpenFont("ComicNeue-Regular.ttf", size);
	SDL_Color foregroundColor = {255, 255, 255};
	SDL_Surface* textSurface = TTF_RenderText_Solid(font, text, foregroundColor);
        SDL_Texture * texture = SDL_CreateTextureFromSurface(renderer, textSurface);
	
	int texW = 0;
	int texH = 0;
	SDL_QueryTexture(texture, NULL, NULL, &texW, &texH);
	SDL_Rect dstrect = {centered?x - texW / 2:x,  y - texH / 2, texW, texH};
	SDL_RenderCopy(renderer, texture, NULL, &dstrect);
	SDL_RenderPresent(renderer);

	SDL_DestroyTexture(texture);
	SDL_FreeSurface(textSurface);
	TTF_CloseFont(font);
}

void drawGridAndLabels(){
	double poscounter = 0;
	
	while (distanceToX(poscounter) < WIDTH){
		drawVertLine(distanceToX(poscounter));
		poscounter += XMINOR;
	}
	
	drawHorizLine(HEIGHT / 2);
	double velcounter = YMINOR;
	
	while (velocityToY(velcounter) < HEIGHT ){
		drawHorizLine(velocityToY(velcounter));
		drawHorizLine(HEIGHT - velocityToY(velcounter));
		velcounter += YMINOR;
	}
	
	double labelpos = XMAJOR;
	double labelvel = YMAJOR;
	while (distanceToX(labelpos) < WIDTH){
		char label[100];
		sprintf(label, "%.0f m", labelpos);
		drawText( distanceToX(labelpos) , HEIGHT - 20, label, true);
		labelpos += XMAJOR;
	}	
	drawText(20, HEIGHT / 2, "0 m/s", false);
	while (velocityToY(labelvel) < HEIGHT){
		char label1[100];
		char label2[100];
		sprintf(label1, "%.1f m/s", labelvel);
		sprintf(label2, "-%.1f m/s", labelvel);
		drawText (10, HEIGHT - velocityToY(labelvel), label1, false);
		drawText (10, velocityToY(labelvel), label2, false);
		labelvel += YMAJOR;
	}
}

void * fftCompute(void * arg) {
    while (true) {
        while (fftctr == prevfftctr) {};

        pthread_mutex_lock( & fft_start_lock);

        in = & fft_in;
        out = & fft_out;
        printf("calculating fft...\n");

        fftw_execute(p);
        prevfftctr = fftctr;
        printf("FFT calc done...\n");
        pthread_mutex_unlock( & fft_start_lock);
        convertToMagSq(fft_out);
        drawToScreen(fft_out);
	drawGridAndLabels();
   }
}

void window1d(double x[], int N) {
    for (int i = 0; i < N; i++) {
        double u = sin(M_PI * i / (N - 1));
        x[i] = u * u;
    }
}

void window2d(double x[FFTSIZE][TIMESIZE]) {
    double row[FFTSIZE];
    double col[TIMESIZE];
    window1d(row, FFTSIZE);
    window1d(col, TIMESIZE);
    for (int i = 0; i < FFTSIZE; i++) {
        for (int j = 0; j < TIMESIZE; j++) {
            x[i][j] = row[i] * col[j];
        }
    }
}

void func(int sockfd) {
    int16_t buff[PSIZE * 2];
    int16_t samples_real[FFTSIZE][TIMESIZE];
    int16_t samples_imag[FFTSIZE][TIMESIZE];
    int n;
    int init = 0;
    int ramp = 0;
    int scnt = 0;
    int firstzero = 0;
    int timeind = 0;
    for (;;) {
        recv(sockfd, buff, sizeof(buff), MSG_WAITALL);
        for (int i = 0; i < PSIZE * 2; i += 2) {
            int16_t s1 = buff[i];
            int16_t s2 = buff[i + 1];
            //printf("%d\n", init);
            if (init) {
                if ((s1 & MASK) == 0) {
                    if (scnt != 0) {
                        if (scnt < SAMPLELOWERBOUND) {
				goto CONTINUEWRITE;
                           printf("Received Size: %d\n", scnt);
                        }
			for (int j = 0; j < ZEROSAMPLES; j++){
				samples_real[j][timeind] = 0;
				samples_imag[j][timeind] = 0;
			}
                        timeind++;
                    }
                    ramp = 1;
                    scnt = 0;
                    if (firstzero && timeind >= TIMESIZE) {
                        timeind = 0;
                        //printf("eggert\n");
                        pthread_mutex_lock( & fft_start_lock);
			
			double averages_real[TIMESIZE];
			double averages_imag[TIMESIZE];

			for (int j = 0 ; j < FFTSIZE; j++){
			    for (int k = 0 ; k < TIMESIZE ; k++){
			        averages_real[k] += samples_real[j][k];
				averages_imag[k] += samples_imag[j][k];
			    }
			}
			
			for (int j = 0 ; j < TIMESIZE; j++){
				averages_real[j] /= FFTSIZE;
				averages_imag[j] /= FFTSIZE;
			}

                        for (int j = 0; j < FFTSIZE; j++) {
                            for (int k = 0; k < TIMESIZE; k++) {
				fft_in[j][k][0] = (samples_real[j][k] - averages_real[k]) * hann[j][k]
											  ;
                                fft_in[j][k][1] = (samples_imag[j][k] - averages_imag[k]) * hann[j][k]
											  ;
                            }
                        }
                        pthread_mutex_unlock( & fft_start_lock);
                        fftctr++;
                        //create stuff to do FFT
                        firstzero = 0;
                    }
                } else {
			CONTINUEWRITE:if (ramp && timeind == 0) {
                        ramp = 0;
                        firstzero = 1;
                        memset(samples_real, 0, sizeof(samples_real));
                        memset(samples_imag, 0, sizeof(samples_imag));
                    }
                    if (scnt < FFTSIZE) {
                        samples_real[scnt][timeind] = (int16_t)(s1 & NMASK);
                        samples_imag[scnt][timeind] = (int16_t)(s2 & NMASK);
                        scnt++;
                    }
                }

            } else {
                //printf("eggert123\n");
                if ((s1 & MASK) == 0) {
                    init = 1;
                    ramp = 1;
                    scnt = 0;
                    printf("Initialized\n");
                }
            }
        }
    }
}


int main() {

    window2d(hann);
	
    printf("Window generated\n");

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(WIDTH , HEIGHT, 0, & window, & renderer);
    SDL_SetWindowTitle( window, "Radar Range and Velocity Plot"); 
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    TTF_Init();
    drawGridAndLabels();
    if (!fftw_init_threads()) {
        printf("Error with threads\n");
        return;
    }
    fftw_plan_with_nthreads(6);
    pthread_t th1;
    pthread_create( & th1, NULL, fftCompute, NULL);
    printf("thread created\n");
    in = & fft_in;
    out = & fft_out;
    p = fftw_plan_dft_2d(FFTSIZE, TIMESIZE, in, out, FFTW_FORWARD, FFTW_MEASURE);
    printf("plan created\n");
    if (pthread_mutex_init( & fft_start_lock, NULL) != 0) {
        printf("\n mutex init failed\n");
        return 1;
    }
    if (pthread_mutex_init( & fft_stop_lock, NULL) != 0) {
        printf("\n mutex init failed\n");
        return 1;
    }

    int sockfd, connfd;
    struct sockaddr_in servaddr, cli;

    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    } else
        printf("Socket successfully created..\n");
    bzero( & servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("10.42.0.149");
    servaddr.sin_port = htons(PORT);

    // connect the client socket to server socket
    if (connect(sockfd, (SA * ) & servaddr, sizeof(servaddr)) !=
        0) {
        printf("connection with the server failed...\n");
        exit(0);
    } else
        printf("connected to the server..\n");

    // function for chat
    func(sockfd);

    // close the socket
    close(sockfd);
}


