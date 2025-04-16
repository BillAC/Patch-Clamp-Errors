import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
Rm = 60e6     # Membrane resistance
Cm = 30e-12   # Membrane capacitance (30 pF)
Rs_values = [0, 12e6]  # Rs = 0 Ω and Rs = 12 MΩ
labels = ['Rs = 0 Ω', 'Rs = 12 MΩ']

# Time setup
t_start = 0
t_end = 70e-3  # 70 ms
t_eval = np.linspace(t_start, t_end, 10000)

# Vcmd step values from -60 mV to +60 mV
Vcmd_steps_mV = np.linspace(-60e-3, 60e-3, 9)  # 9 steps
colors = plt.cm.viridis(np.linspace(0, 1, len(Vcmd_steps_mV)))

# Plot setup
fig, axs = plt.subplots(2, 3, figsize=(16, 10), sharex=True)

for j, Rs in enumerate(Rs_values):
    for i, step in enumerate(Vcmd_steps_mV):
        # Define Vcmd function
        def Vcmd(t, step=step):
            if t < 10e-3:
                return -80e-3
            elif t < 60e-3:
                return step
            else:
                return -80e-3

        time = t_eval
        time_ms = time * 1e3

        if Rs == 0:
            # Vm is instantly equal to Vcmd
            Vcmd_values = np.array([Vcmd(t) for t in time])
            Vm = Vcmd_values
            dVcmd_dt = np.gradient(Vcmd_values, time)
            I_Cm = Cm * dVcmd_dt
            I_Rm = Vm / Rm
        else:
            # ODE for Vm with Rs
            def dVm_dt(t, Vm):
                Vc = Vcmd(t)
                I_rs = (Vc - Vm) / Rs
                I_rm = Vm / Rm
                return (I_rs - I_rm) / Cm

            sol = solve_ivp(dVm_dt, [t_start, t_end], [0], t_eval=time, method='RK45')
            Vm = sol.y[0]
            Vcmd_values = np.array([Vcmd(t) for t in time])
            I_Cm = Cm * np.gradient(Vm, time)
            I_Rm = Vm / Rm

        # Convert to plotting units
        Vm_mV = Vm * 1e3
        Vcmd_mV = Vcmd_values * 1e3
        I_Cm_pA = I_Cm * 1e12
        I_Rm_pA = I_Rm * 1e12
        I_tot_pA = I_Cm_pA + I_Rm_pA

        # Plot each trace
        axs[j, 0].plot(time_ms, Vcmd_mV, color=colors[i], label=f'{step*1e3:.0f} mV')
        axs[j, 0].set_ylim(-100, 75)
        axs[j, 1].plot(time_ms, Vm_mV, color=colors[i])
        axs[j, 1].set_ylim(-100, 75)
        axs[j, 2].plot(time_ms, I_tot_pA, color=colors[i])
        axs[j, 2].set_ylim(-2000, 2000)

# Label and layout
for j, label in enumerate(labels):
    axs[j, 0].set_ylabel(label + '\nVcmd (mV)')
    axs[j, 0].grid(True)
    axs[j, 1].set_ylabel('Vm (mV)')
    axs[j, 1].grid(True)
    axs[j, 2].set_ylabel('I_Total (pA)')
    axs[j, 2].grid(True)

for ax in axs[1, :]:
    ax.set_xlabel('Time (ms)')

# Titles
axs[0, 0].set_title('Vcmd Over Time')
axs[0, 1].set_title('Membrane Voltage Vm')
axs[0, 2].set_title('Membrane Current')


plt.tight_layout()
plt.show()
