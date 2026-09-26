#include "Clock_Ip.h"
#include "OsIf.h"
#include "Emios_Pwm_Ip.h"
#include "Emios_Mcl_Ip.h"
#include "Siul2_Port_Ip.h"
#include "Lpuart_Uart_Ip.h"
#include "IntCtrl_Ip.h"
#include "Siul2_Icu_Ip.h"
#include "Siul2_Dio_Ip.h"
#include "Pit_Ip.h"
#include "Adc_Sar_Ip.h"
#include "nn_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


/* Macros de eMIOS PWM */
#define INSTANCE_0        ((uint8)1U)
#define CHANNEL_4         ((uint8)4U) /* PTA14 - Avanzar */
#define CHANNEL_3         ((uint8)3U) /* PTA13 - Retroceder */

/* Macros de UART */
#define UART_INSTANCE     6U
#define BUFFER_SIZE       256U
#define RX_BUFFER_SIZE 32

/* Macros de Siul2_Port */
#define SIUL2_CONFIG      g_pin_mux_InitConfigArr_PortContainer_0_BOARD_InitPeripherals
#define SIUL2_PINS        NUM_OF_CONFIGURED_PINS_PortContainer_0_BOARD_InitPeripherals

/* Canales de Potenciómetros ADC */
#define ADC_SAR_POT1      0U
#define ADC_SAR_POT2      1U
#define ADC_SAR_POT3      2U
#define ADC_RAW_MAX_REAL  2660.0f
#define ADC_FILTER_ALPHA  0.15f

/* Parámetros Cinemáticos del Encoder */
#define ENCODER_CPR       3332.0f
#define SAMPLE_TIME_S     0.01f   /* 10 ms (100 Hz) */
#define RPM_CONVERSION_FACTOR ((60.0f / ENCODER_CPR) / SAMPLE_TIME_S)
#define TAU_REF_FILTER    0.2f

/* Límites de PWM del eMIOS */
#define PWM_MAX           1000.0f
#define PWM_MIN          -1000.0f

/* Variables ADC */
volatile boolean notif_triggered = FALSE;
volatile uint16 pot1, pot2, pot3;
float fAdcpot1Filtered = 0.0f, fAdcpot2Filtered = 0.0f, fAdcpot3Filtered = 0.0f;

/* Red Neuronal */
volatile uint8_t Detected_Class = 0U;

/* Variables globales del Encoder */
volatile int32_t  Encoder_Position = 0;
volatile uint8_t  Encoder_Direction = 0;
static volatile uint8_t u8LastState = 0U;

/* Estructura PID */
typedef struct {
    float Kp;
    float Ki;
    float Kd;
    float integral;
    float prev_error;
} PIDController;

PIDController pid_speed = {
    .Kp = 5.0f,
    .Ki = 2.f,
    .Kd = 0.1f,
    .integral = 0.0f,
    .prev_error = 0.0f
};

/* Variables de control */
volatile float Target_RPM_User = 0.0f;
volatile float Target_RPM_Filtered = 0.0f;
volatile float Current_RPM = 0.0f;
volatile float Last_Command_PWM = 0.0f;
static int32_t last_position = 0;
static const float ALPHA_REF = SAMPLE_TIME_S / (TAU_REF_FILTER + SAMPLE_TIME_S);

/* Buffers UART */
volatile uint8 BufferIdx = 0U;
uint8 au8Buffer[BUFFER_SIZE];
uint8 au8TxBuffer[BUFFER_SIZE];
volatile uint8 bRxFlag = 0U;
static volatile uint8_t bTxBusy = 0U;
uint8_t rx_byte = 0;
char rx_line[RX_BUFFER_SIZE];
uint8_t rx_idx = 0;
/* Table de Velocidades por Clase */
static const float CLASS_RPM_MAP[5] = {
    0.0f,    /* Clase 0: STOP */
    150.0f,  /* Clase 1: Avance Rápido */
    50.0f,   /* Clase 2: Avance Lento */
   -150.0f,  /* Clase 3: Reversa Rápida */
   -50.0f    /* Clase 4: Reversa Lenta */
};

/* Prototipeo */
void ENCODER_A(void);
void ENCODER_B(void);
void PIT_Callback(void);
void AdcEndOfChainNotif(void);
static inline void Process_Encoder_Quadrature(void);
float Compute_PID(PIDController *pid, float setpoint, float measured);
void Set_Motor_PWM(float command_pwm);
void Speed_Control_Task(void);
void Process_Rx_Command(char* cmd);
void Uart_Callback(const uint8 HwInstance, const Lpuart_Uart_Ip_EventType Event, const void *UserData);
void Send_Telemetry(void);
uint8_t NN_Forward(float p1, float p2, float p3);
void Read_And_Filter_ADC(void);

static inline uint8_t Read_Pin_A_Fast(void) { return (uint8_t)Siul2_Dio_Ip_ReadPin(Canal_A_PORT, Canal_A_PIN); }
static inline uint8_t Read_Pin_B_Fast(void) { return (uint8_t)Siul2_Dio_Ip_ReadPin(Canal_B_PORT, Canal_B_PIN); }

/* ============================================================================
 * EVALUACIÓN DE LA RED NEURONAL (FORWARD PASS: 3 -> 8 -> 5)
 * ============================================================================ */
uint8_t NN_Forward(float p1, float p2, float p3)
{
    float inputs[3] = { p1, p2, p3 };
    float hidden[8];
    float outputs[5];

    /* 1. Capa de Entrada -> Capa Oculta (con Activación ReLU) */
    for (uint8_t j = 0; j < NN_NUM_HIDDEN; j++)
    {
        float sum = b1[j];
        for (uint8_t i = 0; i < NN_NUM_INPUTS; i++)
        {
            sum += inputs[i] * W1[i][j];
        }
        /* Función de activación ReLU: max(0, sum) */
        hidden[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    /* 2. Capa Oculta -> Capa de Salida */
    for (uint8_t k = 0; k < NN_NUM_OUTPUTS; k++)
    {
        float sum = b2[k];
        for (uint8_t j = 0; j < NN_NUM_HIDDEN; j++)
        {
            sum += hidden[j] * W2[j][k];
        }
        outputs[k] = sum;
    }

    /* 3. Selección Argmax (Clase con la salida máxima) */
    uint8_t max_class = 0;
    float max_val = outputs[0];
    for (uint8_t k = 1; k < NN_NUM_OUTPUTS; k++)
    {
        if (outputs[k] > max_val)
        {
            max_val = outputs[k];
            max_class = k;
        }
    }

    return max_class;
}

/* ============================================================================
 * LECTURA Y FILTRADO ADC
 * ============================================================================ */
void AdcEndOfChainNotif(void)
{
    pot1 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_POT1);
    pot2 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_POT2);
    pot3 = Adc_Sar_Ip_GetConvData(ADCHWUNIT_0_INSTANCE, ADC_SAR_POT3);
    notif_triggered = TRUE;
}

void Read_And_Filter_ADC(void)
{
    uint32_t timeout = 10000U;

    notif_triggered = FALSE;
    Adc_Sar_Ip_StartConversion(ADCHWUNIT_0_INSTANCE, ADC_SAR_IP_CONV_CHAIN_NORMAL);

    /* Esperar con Timeout */
    while ((notif_triggered != TRUE) && (timeout > 0U))
    {
        timeout--;
    }

    if (notif_triggered == TRUE)
    {
        notif_triggered = FALSE;

        float p1 = ((float)pot1) / ADC_RAW_MAX_REAL;
        float p2 = ((float)pot2) / ADC_RAW_MAX_REAL;
        float p3 = ((float)pot3) / ADC_RAW_MAX_REAL;

        /* Filtro Pasa Bajas */
        fAdcpot1Filtered = (ADC_FILTER_ALPHA * p1) + ((1.0f - ADC_FILTER_ALPHA) * fAdcpot1Filtered);
        fAdcpot2Filtered = (ADC_FILTER_ALPHA * p2) + ((1.0f - ADC_FILTER_ALPHA) * fAdcpot2Filtered);
        fAdcpot3Filtered = (ADC_FILTER_ALPHA * p3) + ((1.0f - ADC_FILTER_ALPHA) * fAdcpot3Filtered);
    }
}

/* ============================================================================
 * MAIN Y LOOP DE EVALUACIÓN
 * ============================================================================ */
int main(void)
{
    uint32_t u32TxTimer = 0U;

    Clock_Ip_Init(Clock_Ip_aClockConfig);
    Clock_Ip_InitClock(Clock_Ip_aClockConfig);
    while(CLOCK_IP_PLL_LOCKED != Clock_Ip_GetPllStatus());
    Clock_Ip_DistributePll();
    OsIf_Init(NULL_PTR);

    IntCtrl_Ip_Init(&IntCtrlConfig_0);
    IntCtrl_Ip_EnableIrq(LPUART6_IRQn);
    IntCtrl_Ip_EnableIrq(SIUL_0_IRQn);
    IntCtrl_Ip_EnableIrq(PIT0_IRQn);
    IntCtrl_Ip_EnableIrq(ADC0_IRQn);

    Siul2_Port_Ip_Init(SIUL2_PINS, SIUL2_CONFIG);
    u8LastState = (Read_Pin_A_Fast() << 1U) | Read_Pin_B_Fast();

    Siul2_Icu_Ip_Init(0, &Siul2_Icu_Ip_0_Config_PB);
    Siul2_Icu_Ip_EnableInterrupt(0, 1);
    Siul2_Icu_Ip_EnableInterrupt(0, 2);
    Siul2_Icu_Ip_EnableNotification(0, 1);
    Siul2_Icu_Ip_EnableNotification(0, 2);

    Lpuart_Uart_Ip_Init(UART_INSTANCE, &Lpuart_Uart_Ip_xHwConfigPB_6);
    Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, &au8Buffer[0], 1U);

    Emios_Mcl_Ip_Init(INSTANCE_0, &Emios_Mcl_Ip_Sa_1_Config);
    Emios_Pwm_Ip_InitChannel(EMIOS_PWM_IP_SA_I1_CH4_CFG, &Emios_Pwm_Ip_Sa_I1_Ch4);
    Emios_Pwm_Ip_InitChannel(EMIOS_PWM_IP_SA_I1_CH3_CFG, &Emios_Pwm_Ip_Sa_I1_Ch3);
    Set_Motor_PWM(0.0f);

    Adc_Sar_Ip_Init(ADCHWUNIT_0_INSTANCE, &AdcHwUnit_0);
    for (uint8 Index = 0; Index <= 5; Index++) {
        Adc_Sar_Ip_DoCalibration(ADCHWUNIT_0_INSTANCE);
    }
    Adc_Sar_Ip_EnableNotifications(ADCHWUNIT_0_INSTANCE, ADC_SAR_IP_NOTIF_FLAG_NORMAL_ENDCHAIN);

    Pit_Ip_Init(0, &PIT_0_InitConfig_PB);
    Pit_Ip_InitChannel(0, PIT_0_CH_2);
    Pit_Ip_EnableChannelInterrupt(0, 2);
    Pit_Ip_StartChannel(0, 2, 300000U);

    while (1)
        {
    	/* 1. Atender comandos UART si hay una trama completa recibida */
			if (bRxFlag == 1U)
			{
				// CORRECCIÓN 2: Procesar au8Buffer en lugar de rx_line
				Process_Rx_Command((char *)au8Buffer);

				/* Limpieza del buffer */
				memset(au8Buffer, 0, sizeof(au8Buffer));
				BufferIdx = 0U;
				bRxFlag = 0U;

				/* Reiniciar la escucha desde la posición 0 */
				Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, &au8Buffer[0], 1U);
			}

			/* 2. Lectura y filtrado del ADC */
			Read_And_Filter_ADC();

			/* 3. Inferencia de la Red Neuronal */
			Detected_Class = NN_Forward(fAdcpot1Filtered, fAdcpot2Filtered, fAdcpot3Filtered);

			/* 4. Asignación de Setpoint por Clase */
			Target_RPM_User = CLASS_RPM_MAP[Detected_Class];

			/* 5. Envío de Telemetría periódica */
			u32TxTimer++;
			if (u32TxTimer >= 1000U)
			{
				Send_Telemetry();
				u32TxTimer = 0U;
			}
		}

        return 0U;
}

/* ============================================================================
 * LAZO DE CONTROL CONTROL PID (PIT ISR 100 Hz)
 * ============================================================================ */
void PIT_Callback(void)
{
    Speed_Control_Task();
}

void Speed_Control_Task(void)
{
    int32_t current_pos = Encoder_Position;
    int32_t delta_pos = current_pos - last_position;
    last_position = current_pos;

    Current_RPM = (float)delta_pos * RPM_CONVERSION_FACTOR;

    /* Filtrado suave del setpoint asignado por la RN */
    Target_RPM_Filtered = (ALPHA_REF * Target_RPM_User) + ((1.0f - ALPHA_REF) * Target_RPM_Filtered);

    Last_Command_PWM = Compute_PID(&pid_speed, Target_RPM_Filtered, Current_RPM);
    Set_Motor_PWM(Last_Command_PWM);
}

float Compute_PID(PIDController *pid, float setpoint, float measured)
{
    float error = setpoint - measured;

    float p_term = pid->Kp * error;
    pid->integral += error * SAMPLE_TIME_S;

    float max_i_limit = 800.0f;
    if (pid->integral > max_i_limit)  pid->integral = max_i_limit;
    if (pid->integral < -max_i_limit) pid->integral = -max_i_limit;

    float i_term = pid->Ki * pid->integral;
    float derivative = (error - pid->prev_error) / SAMPLE_TIME_S;
    float d_term = pid->Kd * derivative;

    pid->prev_error = error;

    float output = p_term + i_term + d_term;

    if (output > PWM_MAX)       output = PWM_MAX;
    else if (output < PWM_MIN)  output = PWM_MIN;

    return output;
}

void Set_Motor_PWM(float command_pwm)
{
    if (command_pwm > 0.0f)
    {
        Emios_Pwm_Ip_SetDutyCycle(INSTANCE_0, CHANNEL_3, 0U);
        Emios_Pwm_Ip_SetDutyCycle(INSTANCE_0, CHANNEL_4, (uint16_t)command_pwm);
    }
    else if (command_pwm < 0.0f)
    {
        Emios_Pwm_Ip_SetDutyCycle(INSTANCE_0, CHANNEL_4, 0U);
        Emios_Pwm_Ip_SetDutyCycle(INSTANCE_0, CHANNEL_3, (uint16_t)(-command_pwm));
    }
    else
    {
        Emios_Pwm_Ip_SetDutyCycle(INSTANCE_0, CHANNEL_4, 0U);
        Emios_Pwm_Ip_SetDutyCycle(INSTANCE_0, CHANNEL_3, 0U);
    }
}

/* ============================================================================
 * TELEMETRÍA Y COMUNICACIÓN UART
 * ============================================================================ */
void Send_Telemetry(void)
{
    if (bTxBusy == 1U) return;

    /* Trama de 11 valores:
         * P1, P2, P3, Class_Det, Target_User, Target_Filt, Measured_RPM, PWM, Kp, Ki, Kd
         */
	int len = sprintf(
		(char *)au8TxBuffer,
		"%.3f,%.3f,%.3f,%d,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f\r\n",
		(double)fAdcpot1Filtered,
		(double)fAdcpot2Filtered,
		(double)fAdcpot3Filtered,
		Detected_Class,
		(double)Target_RPM_User,
		(double)Target_RPM_Filtered,
		(double)Current_RPM,
		(double)Last_Command_PWM,
		(double)pid_speed.Kp,
		(double)pid_speed.Ki,
		(double)pid_speed.Kd
	);

    bTxBusy = 1U;
    Lpuart_Uart_Ip_AsyncSend(UART_INSTANCE, au8TxBuffer, (uint32)len);
}

/* ============================================================================
 * PROCESAMIENTO DE COMANDOS UART (FUERA DE ISR)
 * ============================================================================ */
void Process_Rx_Command(char *cmd)
{
    float val = 0.0f;
    char *ptr = NULL;


    /* Buscar subcadenas dentro del buffer ignorando ruido inicial */
    if ((ptr = strstr(cmd, "KP:")) != NULL)
    {
        if (sscanf(ptr, "KP:%f", &val) == 1)
        {
            pid_speed.Kp = val;

        }
    }
    else if ((ptr = strstr(cmd, "KI:")) != NULL)
    {
        if (sscanf(ptr, "KI:%f", &val) == 1)
        {
            pid_speed.Ki = val;
            pid_speed.integral = 0.0f; /* Limpiar windup acumulado */

        }
    }
    else if ((ptr = strstr(cmd, "KD:")) != NULL)
    {
        if (sscanf(ptr, "KD:%f", &val) == 1)
        {
            pid_speed.Kd = val;

        }
    }
}

/* ==============================================================================
 * CALLBACK UART INTERRUPT / EVENT (NXP RTD)
 * ============================================================================== */
void Uart_Callback(
    const uint8 HwInstance,
    const Lpuart_Uart_Ip_EventType Event,
    const void *UserData)
{
    (void)UserData;
    (void)HwInstance;

    switch (Event)
    {
        case LPUART_UART_IP_EVENT_RX_FULL:


            /* 1. Verificar fin de trama (\n o \r) */
            if ((au8Buffer[BufferIdx] == '\n') || (au8Buffer[BufferIdx] == '\r'))
            {
                if (BufferIdx > 0U)
                {
                    au8Buffer[BufferIdx] = '\0'; /* Terminar string */
                    bRxFlag = 1U;                /* Avisar al main() */
                }
                else
                {
                    /* Si llega un \n suelto, seguimos escuchando en el índice 0 */
                    Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, &au8Buffer[0], 1U);
                }
            }
            /* 2. Si es un carácter normal, rearmar para el siguiente índice */
            else if (BufferIdx < (BUFFER_SIZE - 2U))
            {
                BufferIdx++;
                /* Mover el puntero de RX según la API de RTD de NXP */
                Lpuart_Uart_Ip_SetRxBuffer(UART_INSTANCE, &au8Buffer[BufferIdx], 1U);
                Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, &au8Buffer[BufferIdx], 1U);
            }
            else
            {
                au8Buffer[BufferIdx] = '\0';
                bRxFlag = 1U;
            }
            break;

        case LPUART_UART_IP_EVENT_ERROR:

            BufferIdx = 0U;
            bRxFlag = 0U;
            Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, &au8Buffer[0], 1U);
            break;

        case LPUART_UART_IP_EVENT_TX_EMPTY:
            bTxBusy = 0U;
            break;

        default:
            break;
    }
}

static inline void Process_Encoder_Quadrature(void)
{
    static const int8_t QuadratureTable[16] = { 0,1,-1,0, -1,0,0,1, 1,0,0,-1, 0,-1,1,0 };
    uint8_t u8CurrentState = (Read_Pin_A_Fast() << 1U) | Read_Pin_B_Fast();
    uint8_t u8Index = (u8LastState << 2U) | u8CurrentState;
    int8_t s8Step = QuadratureTable[u8Index];
    if (s8Step != 0) {
        Encoder_Position += s8Step;
        Encoder_Direction = (uint8_t)(s8Step > 0 ? 1U : 0U);
    }
    u8LastState = u8CurrentState;
}

void ENCODER_A(void) { Process_Encoder_Quadrature(); }
void ENCODER_B(void) { Process_Encoder_Quadrature(); }

