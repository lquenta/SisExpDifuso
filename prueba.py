import os
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Set working directory
os.chdir("D:/Sergio/classes/1.Maestría_IngMat/7. Fuzzy logic and Fuzzy sets/trabajo final/data")

#####################################
# DATA MANAGEMENT
#####################################
df = pd.read_excel('data.xlsx',sheet_name='data4')
gdp = df['yreal'].values
inflation = df['INF'].values  # prueba

#####################################
# DATA ANALYSIS
#####################################
# Plot time series
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['yreal'], marker='o')
plt.title('PIB anual')
plt.xlabel('Year')
plt.ylabel('PIB real')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['INF'], marker='o')
plt.title('Inflación anual (medida por cambio del IPC)')
plt.xlabel('Year')
plt.ylabel('Inflación')
plt.tight_layout()
plt.show()

#####################################
# MODEL
#####################################
####################################################################
# Fuzzy membership functions (Fuzzy c-means and triangular shapes)
####################################################################
def mfs(x):
    data = np.expand_dims(x, axis=0)
    #Clusters=2, sólo para diferenciar entre bajo y alto
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, c=2, m=2.0, error=0.005, maxiter=1000, init=None)
    sorted_indices = np.argsort(cntr[:, 0])
    centers = cntr[sorted_indices, 0]
    x_plot = np.linspace(np.min(x), np.max(x), 1000)
    min=np.min(x)-0.001
    max=np.max(x)+0.001
    low_mf = fuzz.trimf(x_plot, [min, centers[0], centers[1]])
    high_mf = fuzz.trimf(x_plot, [centers[0], centers[1], max])
    return x_plot, low_mf, high_mf, centers
#########################################################
# Calibration of membership functions for input variables
#########################################################
#GDP
x_plot_gdp, gdp_low_mf, gdp_high_mf, gdp_centers = mfs(gdp)
#Inflation
x_plot_inf, inf_low_mf, inf_high_mf, inf_centers = mfs(inflation)
#Gráfica
x = gdp  
x_p = np.linspace(np.min(x), np.max(x), 1000)
plt.figure(figsize=(8, 4))
plt.plot(x_p, gdp_low_mf , label='Low')
plt.plot(x_p, gdp_high_mf, label='High')
plt.title("Two-Cluster, Data-Driven Triangular Membership Functions")
plt.xlabel("ggdpeu")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

x = inflation
x_plot = np.linspace(np.min(x), np.max(x), 1000)
plt.figure(figsize=(8, 4))
x_plot = np.linspace(np.min(x), np.max(x), 1000)
plt.plot(x_plot, inf_low_mf , label='Low')
plt.plot(x_plot, inf_high_mf, label='High')
plt.title("Two-Cluster, Data-Driven Triangular Membership Functions")
plt.xlabel("ggdpeu")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

##############################################################################
# Definition of Output values for interest rate (Interest Rate from 0% to 10%)
##############################################################################
# x_output is interest rate change; get higher, get lower, or no change.
# grid interval to calculate final numerical value
x_output = np.linspace(-10, 10, 1000)
ir_low = fuzz.trimf(x_output, [-10, -5, 0])
ir_none = fuzz.trimf(x_output, [-5, 0, 5])
ir_high = fuzz.trimf(x_output, [0, 5, 10])

##############################################################################
# APPLICATION
##############################################################################
# Input data point
input_gdp = gdp[9]
input_inf = inflation[9]

# Calculando pertenencia a cada intérvalo para input
x = np.linspace(np.min(gdp), np.max(gdp), 1000)
input_gdp_gdp_low_mf_mfv= fuzz.interp_membership(x, gdp_low_mf, input_gdp)
input_gdp_gdp_high_mf_mfv= fuzz.interp_membership(x, gdp_high_mf, input_gdp)
x = np.linspace(np.min(inflation), np.max(inflation), 1000)
input_inf_inf_low_mf_mfv= fuzz.interp_membership(x, inf_low_mf, input_inf)
input_inf_inf_high_mf_mfv= fuzz.interp_membership(x, inf_high_mf, input_inf)

##############################################################################
# Rules for inference
##############################################################################
# Rule 1: IF GDP growth is Low AND Inflation is Low THEN Interest Rate is Low
# Rule 2: IF GDP growth is Low AND Inflation is High THEN Interest Rate is none
# Rule 3: IF GDP growth is High AND Inflation is Low THEN Interest Rate is none
# Rule 4: IF GDP growth is High AND Inflation is High THEN Interest Rate is Low

##############################################################################
# Output for Rules for inference
##############################################################################
#For each condition
rule1=min(input_gdp_gdp_low_mf_mfv,input_inf_inf_low_mf_mfv)
rule2=min(input_gdp_gdp_low_mf_mfv,input_inf_inf_high_mf_mfv)
rule3=min(input_gdp_gdp_high_mf_mfv,input_inf_inf_low_mf_mfv)
rule4=min(input_gdp_gdp_high_mf_mfv,input_inf_inf_high_mf_mfv)

#Truncation for each rule
rule1_output = np.fmin(rule1, ir_low)
rule2_output = np.fmin(rule2, ir_none)
rule3_output = np.fmin(rule3, ir_none)
rule4_output = np.fmin(rule4, ir_high)

##############################################################################
# Intermediate fuzzy Output: Aggregation through output
##############################################################################
# Maximum
aggregated = np.fmax(rule1_output,np.fmax(rule2_output, np.fmax(rule3_output, rule4_output)))

##############################################################################
# Final numerical Output: Centroid
##############################################################################
interest_rate_result = fuzz.defuzz(x_output, aggregated, 'centroid')

##############################################################################
# FINAL RESULTS OUTPUT
##############################################################################
# Gráfico
plt.figure(figsize=(8, 4))
plt.plot(x_output, ir_low, 'b--', alpha=0.5, label='Low IR MF')
plt.plot(x_output, ir_high, 'r--', alpha=0.5, label='High IR MF')
plt.plot(x_output, ir_none, 'g--', alpha=0.5, label='None IR MF')
plt.fill_between(x_output, aggregated, alpha=0.7, label='Aggregated Output')
plt.axvline(interest_rate_result, color='k', linestyle='--', label=f'Crisp Output: {interest_rate_result:.2f}%')
plt.title("Fuzzy Inference Result: Interest Rate")
plt.xlabel("Interest Rate (%)")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

# Resultados
print("==== INPUTS ====")
print(f"GDP: {input_gdp:.2f}, Inflation: {input_inf:.2f}")
print("==== MEMBERSHIPS ====")
print(f"GDP Low: {input_gdp_gdp_low_mf_mfv:.2f}, GDP High: {input_gdp_gdp_high_mf_mfv:.2f}")
print(f"Inflation Low: {input_inf_inf_low_mf_mfv:.2f}, Inflation High: {input_inf_inf_high_mf_mfv:.2f}")
print("==== RULE ACTIVATIONS ====")
print(f"Rule 1 (Low, Low): {rule1:.2f}")
print(f"Rule 2 (Low, High): {rule2:.2f}")
print(f"Rule 3 (High, Low): {rule3:.2f}")
print(f"Rule 4 (High, High): {rule4:.2f}")
print("==== OUTPUT ====")
print(f"Interest Rate: {interest_rate_result:.2f}%")