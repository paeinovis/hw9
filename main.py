import stat
import statistics

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate

# 1
# Used this because I don't know Basic Math lol
# https://www.mathsisfun.com/data/least-squares-regression.html

CO_vals = np.genfromtxt('ks_observables.dat', skip_header=1, usecols=0)
SF_vals = np.genfromtxt('ks_observables.dat', skip_header=1, usecols=1)

CO_vals = np.log10(CO_vals)     # x
SF_vals = np.log10(SF_vals)     # y

xys = CO_vals * SF_vals
x_2 = CO_vals ** 2

size_ = np.size(CO_vals)

slope_ = (size_ * np.sum(xys) - ((np.sum(CO_vals)) * (np.sum(SF_vals)))) / (size_ * (np.sum(x_2)) - ((np.sum(CO_vals)) ** 2))

b_ = (np.sum(SF_vals) - (slope_ * np.sum(CO_vals))) / size_

x_s = np.linspace(min(CO_vals), max(CO_vals), size_)
y_s = slope_ * x_s + b_

plt.scatter(CO_vals, SF_vals)
plt.plot(x_s, y_s, color="black")
plt.xlim(np.amin(CO_vals), np.amax(CO_vals))
plt.xlabel("CO Brightness")
plt.ylabel("SF Rate")
plt.title(f"Carbon Monoxide Brightness Compared to Star Formation Rate\nAnd Fitted Line y={round(slope_, 2)}x+({round(b_, 2)})")
# plt.savefig('swanson-hw9.png')
plt.show()

print(f"Linear least squares fit: y = {round(slope_, 2)}x + ({round(b_, 2)})\n")



# 2
# This (HII regions) is like literally what I'm supposed to be getting paid by UF to work on
# I'm embarrassed that it's taken me this long to actually code anything related to it

def gauss(x, A, B, C):
    return A * np.exp(-((x - B) ** 2) / (2 * C ** 2))

# a
vel_a = np.genfromtxt('hw10a.dat', usecols=0)
temp_a = np.genfromtxt('hw10a.dat', usecols=1)

ssdev = statistics.stdev(vel_a.tolist())

init_vals = [np.amax(temp_a), np.amax(vel_a), ssdev]   # a = height of peak, b = location of center of peak, ssdev = standard deviation
par, cov = curve_fit(gauss, vel_a, temp_a, p0=init_vals)
fit_ys = gauss(vel_a, par[0], par[1], par[2])
fit_ys_2 = gauss(vel_a, np.amax(temp_a), par[1], par[2])

plt.scatter(vel_a, temp_a)
plt.plot(vel_a, fit_ys, color="black")
plt.plot(vel_a, fit_ys_2, color="orange")
# First (black) is fitted but second (orange) is with original max height
# because otherwise it doesn't really reach the peak
plt.xlim(np.amin(vel_a), np.amax(vel_a))
plt.xlabel("Velocity (km/s)")
plt.ylabel("Temperature")
plt.title("Guassian for Emission Line of Data a\nAnd Fitted Gaussian")
# plt.savefig('swanson-hw9a.png')
plt.show()


# b
# FWHM (x where f(x) = y/2) * 2; using the first (black) curve
y_max = np.amax(fit_ys)
x_fwhm_ind = np.abs(fit_ys - (np.amax(fit_ys) / 2)).argmin()
diff_ind = y_max - x_fwhm_ind

x_fwhm = vel_a[x_fwhm_ind]
ys_fwhm = fit_ys[int(x_fwhm_ind):int((y_max + diff_ind))]

area_r = np.trapz(ys_fwhm, dx=.5)
print("Area a: ", area_r)


# c
vel_b = np.genfromtxt('hw10b.dat', usecols=0)
temp_b = np.genfromtxt('hw10b.dat', usecols=1)

ssdev_b = statistics.stdev(vel_b.tolist())

init_vals_b = [np.argmax(temp_b), np.argmax(vel_b), ssdev_b]   # a = height of peak, b = location of center of peak, ssdev = standard deviation
par_b, cov_b = curve_fit(gauss, vel_b, temp_b, p0=init_vals_b)
fit_ys_b = gauss(vel_b, par_b[0], par_b[1], par_b[2])
fit_ys_2_b = gauss(vel_b, np.amax(temp_b), par_b[1], par_b[2])

plt.scatter(vel_b, temp_b)
plt.plot(vel_b, fit_ys_b, color="black")
plt.plot(vel_b, fit_ys_2_b, color="orange")
# First (black) is fitted but second (orange) is with original max height
# because otherwise it doesn't really reach the peak
plt.xlim(np.amin(vel_b), np.amax(vel_b))
plt.xlabel("Velocity (km/s)")
plt.ylabel("Temperature")
plt.title("Gaussian for Emission Line of Data b\nAnd Fitted Single Gaussian")
# plt.savefig('swanson-hw9b.png')
plt.show()


# d
# FWHM (x where f(x) = y/2) * 2; using the first (black) curve
y_max_b = np.amax(fit_ys_b)
x_fwhm_ind_b = np.abs(fit_ys_b - (np.amax(fit_ys_b) / 2)).argmin()
diff_ind_b = int(y_max_b - x_fwhm_ind_b)

x_fwhm_b = vel_b[x_fwhm_ind_b]
ys_fwhm_b = fit_ys_b[int(y_max_b + diff_ind_b):x_fwhm_ind_b]

area_r_b = np.trapz(ys_fwhm_b, dx=.5)
print("Area b:", area_r_b)


# EC
# two diff peaks then add ?
midpoint = int(vel_b.size/2)
vel_b_p1 = vel_b[:midpoint]
temp_b_p1 = temp_b[:midpoint]

vel_b_p2 = vel_b[midpoint:]
temp_b_p2 = temp_b[midpoint:]

ssdev_b_p1 = statistics.stdev(vel_b_p1.tolist())
ssdev_b_p2 = statistics.stdev(vel_b_p2.tolist())

init_vals_p1 = [np.argmax(temp_b_p1), np.argmax(vel_b_p1), ssdev_b_p1]   # a = height of peak, b = location of center of peak, ssdev = standard deviation
init_vals_p2 = [np.argmax(temp_b_p2), np.argmax(vel_b_p2), ssdev_b_p2]   # a = height of peak, b = location of center of peak, ssdev = standard deviation

par_b_p1, cov_b_p1 = curve_fit(gauss, vel_b_p1, temp_b_p1, p0=init_vals_p1)
par_b_p2, cov_b_p2 = curve_fit(gauss, vel_b_p2, temp_b_p2, p0=init_vals_p2)

fit_ys_p1 = gauss(vel_b_p1, par_b_p1[0], par_b_p1[1], par_b_p1[2])
fit_ys_p2 = gauss(vel_b_p2, par_b_p2[0], par_b_p2[1], par_b_p2[2])

tot_array = np.concatenate([fit_ys_p1, fit_ys_p2])

plt.scatter(vel_b, temp_b)
plt.plot(vel_b, tot_array, color="black")
plt.xlim(np.amin(vel_b), np.amax(vel_b))
plt.xlabel("Velocity (km/s)")
plt.ylabel("Temperature")
plt.title("Gaussian for Emission Line of Data a\nAnd Fitted Gaussian, Double Peak")
# plt.savefig('swanson-hw9bec.png')
plt.show()
