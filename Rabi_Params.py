import matplotlib.pyplot as plt
import numpy as np
from arc import *
from matplotlib.widgets import Slider
from numpy import pi
from scipy.constants import epsilon_0, c, physical_constants, hbar

# Parameterbereiche

detuning_lst = 2*pi* np.array([-38,-17,-8])*1e9

# P1_lst = np.linspace(10,160,16)*1e-3
# P2_lst = np.linspace(10,160,16)*1e-3
P1_lst = np.linspace(0.5,5,10)*1e-3
P2_lst = np.linspace(0.5,5,10)*1e-3

d=np.array([43.5,87,130.5,174,217.5,225,235.5,250])*1e-6 #Bis ca. 250um in Breite und Länge bei Adrian Thiel
area_lst = []
for length in d:
    area_lst.append(length*length)


# Physik

rb85 = Rubidium85()

# Laser Parameters
Pa = 165e-3
wa = 1.8e-6
qa = -1
Pb =165e-3
wb = 1.8e-6
qb = -1

f1 = 2
mf1 = 0
f0 = 3
mf0 = 0

# D1 Line
ne = 5
le = 1
je = 0.5


# def two_photon_rabi(Pa, Pb, Area, Delta0, f0=f_start, mf0=mf_start, f1=f_end, mf1=mf_end, ne=n_qn, le=l_end, je=j_end,qa=pol_a,qb=pol_b):
#
#     OmR, AC, Psc = rb85.groundStateRamanTransition(
#         Pa, wa, qa, Pb, wb, qb, Delta0, f0, mf0, f1, mf1, ne, le, je)
#
#     return OmR
def _reducedMatrixElementFJ(
         j1: float, f1: float, j2: float, f2: float) -> float:
    sph = 0.0
    if (abs(f2 - f1) < 2) & (round(abs(j2 - j1)) < 2):
        # Reduced Matrix Element <f||er||f'> in units of reduced matrix element <j||er||j'>
        sph = (
                (-1.0) ** (j1 + rb85.I + f2 + 1.0)
                * ((2.0 * f1 + 1) * (2 * f2 + 1)) ** 0.5
                * Wigner6j(f1, 1, f2, j2, rb85.I, j1)
        )

    return sph
#Berechnung der Rabi-Frequenz über alle möglichen Zwischen-Niveaus, wenn mit einem Flattop-Profil auf das Atom-Raster eingestrahlt wird
def two_photon_rabi_flat(
        Pa, Pb, Delta, Aa, Ab, qa=-1, qb=-1,
        f0=2, mf0=0, f1=3, mf1=0, ne=5, le=1, je=0.5
):
    """
    Berechnet die zwei-photon-Rabi-Frequenz, den differentiellen AC Stark Shift
    und die Wahrscheinlichkeit zum Photonstreuen für Flattop-Strahlen.

    Pa, Pb: Laserleistungen [W]
    Aa, Ab: Flächen der Flattop-Strahlen [m^2]
    qa, qb: Polarisationszeichen (+1,0,-1)
    Delta: Detuning [rad/s]
    f0,mf0,f1,mf1: Qubitzustände
    ne, le, je: angeregter Zustand
    """


    # Flattop Intensität
    Ia = Pa / Aa
    Ib = Pb / Ab

    Ea = np.sqrt(2 * Ia / (epsilon_0 * c))
    Eb = np.sqrt(2 * Ib / (epsilon_0 * c))


    # Grundzustand
    ng = rb85.groundStateN
    lg = 0
    jg = 0.5
    I_nuclear = rb85.I

    rme_j = rb85.getReducedMatrixElementJ(ng, lg, jg, ne, le, je)
    rme_j *= physical_constants["Bohr radius"][0] * physical_constants["elementary charge"][0]

    A, B = rb85.getHFSCoefficients(ng, lg, jg)
    omega01 = (jg + I_nuclear) * A * 2 * pi


    # angeregter Zustand
    A_exc, B_exc = rb85.getHFSCoefficients(ne, le, je)
    Gamma = 1 / rb85.getStateLifetime(ne, le, je)

    OmegaR = 0.0
    AC1 = 0.0
    AC0 = 0.0
    Pe = 0.0


    # Hyperfine-Loops
    for fe in range(round(abs(je - I_nuclear)), round(1 + (je + I_nuclear))):
        Ehfs = 2 * pi * rb85.getHFSEnergyShift(je, fe, A_exc, B_exc)

        for mfe in range(max(-fe, min(mf1, mf0) - 1), 1 + min(fe, max(mf1, mf0) + 1)):
            Omaf0 = (Ea * rme_j / hbar *
                     rb85.getSphericalDipoleMatrixElement(f0, mf0, fe, mfe, qa) *
                     _reducedMatrixElementFJ(jg, f0, je, fe))
            Omaf1 = (Ea * rme_j / hbar *
                     rb85.getSphericalDipoleMatrixElement(f1, mf1, fe, mfe, qa) *
                     _reducedMatrixElementFJ(jg, f1, je, fe))
            Ombf0 = (Eb * rme_j / hbar *
                     rb85.getSphericalDipoleMatrixElement(f0, mf0, fe, mfe, qb) *
                     _reducedMatrixElementFJ(jg, f0, je, fe))
            Ombf1 = (Eb * rme_j / hbar *
                     rb85.getSphericalDipoleMatrixElement(f1, mf1, fe, mfe, qb) *
                     _reducedMatrixElementFJ(jg, f1, je, fe))

            # AC-Stark
            AC1 += Ombf1 ** 2 / (4 * (Delta - Ehfs)) + Omaf1 ** 2 / (4 * (Delta + omega01 - Ehfs))
            AC0 += Omaf0 ** 2 / (4 * (Delta - Ehfs)) + Ombf0 ** 2 / (4 * (Delta - omega01 - Ehfs))

            # Raman Rabi
            OmegaR += Omaf0 * Ombf1 / (2 * (Delta - Ehfs))

            # Excited population
            Pe += (0.5 * Omaf0 ** 2 / (2 * (Delta - Ehfs) ** 2)
                   + 0.5 * Ombf1 ** 2 / (2 * (Delta - Ehfs) ** 2)
                   + 0.5 * Omaf1 ** 2 / (2 * (Delta + omega01 - Ehfs) ** 2)
                   + 0.5 * Ombf0 ** 2 / (2 * (Delta - omega01 - Ehfs) ** 2))

    # Differential AC shift
    AC = AC0 - AC1

    # π-Puls Zeit
    tau_pi = pi / np.abs(OmegaR)

    # Spontane Streuwahrscheinlichkeit
    Psc = Gamma * tau_pi * Pe

    OmegaR=np.abs(OmegaR)
    return OmegaR


# Speicher
metric = np.zeros((len(detuning_lst),len(area_lst),
                   len(P1_lst),len(P2_lst)))

for i,detuning in enumerate(detuning_lst):
    for j,area in enumerate(area_lst):
        for k,P1 in enumerate(P1_lst):
            for l,P2 in enumerate(P2_lst):

                metric[i,j,k,l] = two_photon_rabi_flat(
                    P1,P2,detuning,area,area
                )
metric=metric/1e9
P1_lst=P1_lst*1e3
P2_lst=P2_lst*1e3

# Plot
fig,ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

detuning_index = 2
area_index = 4

image = ax.pcolormesh(
    P1_lst,
    P2_lst,
    metric[detuning_index,area_index],
    shading="auto",
    cmap="viridis"
)

cbar = fig.colorbar(image)
cbar.set_label("Ω₂ph (GHz)")
cbar.formatter.set_scientific(False)
cbar.formatter.set_useOffset(False)
cbar.update_ticks()

ax.set_xlabel("P1 (mW)")
ax.set_ylabel("P2 (mW)")

title = ax.set_title("")


# Marker Speicherung
points = {}


# Annotation
annot = ax.annotate("",
                    xy=(0,0),
                    xytext=(15,15),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round",fc="white"))

annot.set_visible(False)


# Hover
def hover(event):

    if event.inaxes != ax:
        return

    x,y = event.xdata,event.ydata

    if x is None:
        return

    i = np.argmin(abs(P1_lst-x))
    j = np.argmin(abs(P2_lst-y))

    Omega = metric[detuning_index,area_index,i,j]

    text = (
        f"P1 = {P1_lst[i]:.2f} mW\n"
        f"P2 = {P2_lst[j]:.2f} mW\n"
        f"Δ = {detuning_lst[detuning_index]/ (2*pi*1e9):.1f} GHz\n"
        f"A = {area_lst[area_index]*1e12:.2f} um²\n"
        f"Ω₂ph = {Omega*1e3:.3f} MHz"
    )

    annot.xy = (P1_lst[i],P2_lst[j])
    annot.set_text(text)
    annot.set_visible(True)

    fig.canvas.draw_idle()


# Klick Toggle
def click(event):

    if event.inaxes != ax:
        return

    x,y = event.xdata,event.ydata

    i = np.argmin(abs(P1_lst-x))
    j = np.argmin(abs(P2_lst-y))

    key = (i,j)

    if key in points:
        point, box = points[key]
        point.remove()
        box.remove()
        del points[key]

    else:

        point = ax.plot(P1_lst[i],P2_lst[j],"ro")[0]

        Omega = metric[detuning_index,area_index,i,j]

        text = (
                "\nSelected:\n"
                f"P1 = {P1_lst[i]:.3f}mW\n"
                f"P2 = {P2_lst[j]:.3f}mW\n"
                f"Δ = {detuning_lst[detuning_index]/ (2*pi*1e9):.2f} GHz\n"
                f"A = {area_lst[area_index]*1e12:.3f} um²\n"
                f"Ω₂ph = {Omega*1e3:.3f} MHz"
                )
        box=ax.annotate(
            text,xy=(P1_lst[i],P2_lst[j]),xytext=(15,15),textcoords="offset points",bbox=dict(boxstyle="round",fc="white")
        )

        points[key] = (point,box)
    fig.canvas.draw_idle()


# Slider
ax_detuning = plt.axes([0.2,0.15,0.6,0.03])
ax_area = plt.axes([0.2,0.1,0.6,0.03])

slider_detuning = Slider(
    ax_detuning,
    "Detuning",
    0,
    len(detuning_lst)-1,
    valinit=detuning_index,
    valstep=1
)

slider_area = Slider(
    ax_area,
    "Area",
    0,
    len(area_lst)-1,
    valinit=area_index,
    valstep=1
)


# Update Plot
def update(val):

    global detuning_index,area_index

    detuning_index = int(slider_detuning.val)
    area_index = int(slider_area.val)

    # alle markierten Punkte löschen
    for point, box in points.values():
        point.remove()
        box.remove()

    points.clear()
    image.set_array(
        metric[detuning_index,area_index].ravel()
    )

    title.set_text(
        f"Δ = {detuning_lst[detuning_index]/ (2*pi*1e6):.1f} MHz | "
        f"A = {area_lst[area_index]*1e12:.0f} um²"
    )

    fig.canvas.draw_idle()

slider_detuning.on_changed(update)
slider_area.on_changed(update)


# Events
fig.canvas.mpl_connect("motion_notify_event",hover)
fig.canvas.mpl_connect("button_press_event",click)

update(None)

plt.show()