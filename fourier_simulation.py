# ---------------------------------------------------------
# Simulación y análisis de señales usando Transformada de Fourier
# Autor: [Nombre del estudiante]
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parámetros generales
# -------------------------------
fs = 1000  # Frecuencia de muestreo
t = np.arange(-1, 1, 1/fs)  # Vector de tiempo

# -------------------------------
# 1. Señal Pulso Rectangular
# -------------------------------
rect = np.where(np.abs(t) < 0.2, 1, 0)

# -------------------------------
# 2. Señal Escalón
# -------------------------------
step = np.where(t >= 0, 1, 0)

# -------------------------------
# 3. Señal Senoidal
# -------------------------------
f = 5  # frecuencia 5 Hz
sine = np.sin(2 * np.pi * f * t)

# -------------------------------
# Función para calcular FFT
# -------------------------------
def compute_fft(signal):
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    fft_signal = np.fft.fftshift(fft_signal)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    magnitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)
    return freq, magnitude, phase

# -------------------------------
# Cálculo FFT
# -------------------------------
freq_rect, mag_rect, phase_rect = compute_fft(rect)
freq_step, mag_step, phase_step = compute_fft(step)
freq_sine, mag_sine, phase_sine = compute_fft(sine)

# -------------------------------
# Graficar señales en tiempo
# -------------------------------
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(t, rect)
plt.title("Pulso Rectangular")

plt.subplot(3,1,2)
plt.plot(t, step)
plt.title("Función Escalón")

plt.subplot(3,1,3)
plt.plot(t, sine)
plt.title("Señal Senoidal")

plt.tight_layout()
plt.show()

# -------------------------------
# Graficar Magnitud FFT (ejemplo seno)
# -------------------------------
plt.figure()
plt.plot(freq_sine, mag_sine)
plt.title("Magnitud Espectro - Señal Senoidal")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.show()
