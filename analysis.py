
import csv
import matplotlib.pyplot as plt
import numpy as np

file = open('natural_gas_futures_weekly_all.csv','r')
csv_handle = csv.DictReader(file)
weekly_prices = []
dates = []

for rows in csv_handle:
    dates.append(rows['Date'])
    weekly_prices.append(0.5 * (float(rows['High']) + float(rows['Low'])) )
file.close()

plt.plot(range(len(weekly_prices)), weekly_prices, '-b')
plt.xlabel('Week #')
plt.ylabel('Crude Oil Future Price')

from numpy.fft import fft, ifft
from numpy import real,imag

# here we have computed the fft of the weekly_prices
fft_data =  fft(weekly_prices)
N = len(fft_data)
assert(N == len(weekly_prices))
# TODO: first fill in the frequencies call this list 
# fft_frequencies -- it must have length N
# it must store the frequencies of each element in the fft_data
# ensure that the frequencies of the second half are negative.
# your code here

fft_frequencies = []
for k in range(N):
    if k <= N // 2:
        fft_frequencies.append(k / N)
    else:
        fft_frequencies.append((k - N) / N)

print(f"Sample frequencies: {fft_frequencies[:10]} ... {fft_frequencies[-10:]}")
print(f"Frequency range: [{min(fft_frequencies)}, {max(fft_frequencies)}]")

# This function will be useful for you. Please go through the code.

def select_all_items_in_freq_range(lo, hi):
    # TODO: go through the fft_data and select only those frequencies in the range lo/hi
    new_fft_data = [] # make sure we have the 0 frequency component
    for (fft_val, fft_freq) in zip(fft_data, fft_frequencies):
        if lo <= fft_freq and fft_freq < hi:
            new_fft_data.append(fft_val)
        elif -hi < fft_freq and fft_freq <= -lo:
            new_fft_data.append(fft_val)
        else:
            new_fft_data.append(0.0)
    filtered_data = ifft(new_fft_data)
    assert all( abs(imag(x)) <= 1E-10 for x in filtered_data)
    return [real(x) for x in filtered_data]

upto_1_year = select_all_items_in_freq_range(0, 1/52)
one_year_to_1_quarter = select_all_items_in_freq_range(1/52, 1/13)  
less_than_1_quarter = select_all_items_in_freq_range(1/13, 0.5)

# TODO: Redefine the three lists using the select_all_items function
# your code here
print(f"Original data length: {len(weekly_prices)}")
print(f"Up to 1 year components length: {len(upto_1_year)}")
print(f"1 year to 1 quarter components length: {len(one_year_to_1_quarter)}")
print(f"Less than 1 quarter components length: {len(less_than_1_quarter)}")

# Verify that the sum of components equals original (approximately)
reconstructed = np.array(upto_1_year) + np.array(one_year_to_1_quarter) + np.array(less_than_1_quarter)
reconstruction_error = np.mean(np.abs(np.array(weekly_prices) - reconstructed))
print(f"Reconstruction error: {reconstruction_error}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(weekly_prices)
plt.title('Original Weekly Prices')
plt.xlabel('Week #')
plt.ylabel('Price')

plt.subplot(2, 2, 2)
plt.plot(upto_1_year)
plt.title('Long-term trends (> 1 year cycles)')
plt.xlabel('Week #')
plt.ylabel('Price')

plt.subplot(2, 2, 3)
plt.plot(one_year_to_1_quarter)
plt.title('Seasonal patterns (1 year to 1 quarter cycles)')
plt.xlabel('Week #')
plt.ylabel('Price')

plt.subplot(2, 2, 4)
plt.plot(less_than_1_quarter)
plt.title('High frequency noise (< 1 quarter cycles)')
plt.xlabel('Week #')
plt.ylabel('Price')

plt.tight_layout()
plt.show()

plt.plot(upto_1_year,'-b',lw=2)
plt.plot(weekly_prices,'--r',lw=0.2)
plt.xlabel('Week #')
plt.ylabel('Price')
plt.title('Frequency components < once/year')

plt.figure()
plt.plot(one_year_to_1_quarter,'-b',lw=2)
plt.plot(weekly_prices,'--r',lw=0.2)
plt.title('Frequency components between once/year and once/quarter')
plt.xlabel('Week #')
plt.ylabel('Price')

plt.figure()
plt.plot(less_than_1_quarter,'-b',lw=2)
plt.plot(weekly_prices,'--r',lw=0.2)
plt.title('Frequency components >  once/quarter')
plt.xlabel('Week #')
plt.ylabel('Price')

plt.figure()
plt.plot([(v1 + v2 + v3) for (v1, v2, v3) in zip(upto_1_year,one_year_to_1_quarter,less_than_1_quarter)],'-b',lw=2)
plt.plot(weekly_prices,'--r',lw=0.2)
plt.title('Sum of all the components')
plt.xlabel('Week #')
plt.ylabel('Prices')

N = len(weekly_prices)
assert(len(fft_frequencies) == len(weekly_prices))
assert(fft_frequencies[0] == 0.0)
assert(abs(fft_frequencies[N//2] - 0.5 ) <= 0.05), f'fft frequncies incorrect: {fft_frequencies[N//2]} does not equal 0.5'
assert(abs(fft_frequencies[N//4] - 0.25 ) <= 0.05), f'fft frequncies incorrect:  {fft_frequencies[N//4]} does not equal 0.25'
assert(abs(fft_frequencies[3*N//4] + 0.25 ) <= 0.05), f'fft frequncies incorrect:  {fft_frequencies[3*N//4]} does not equal -0.25'
assert(abs(fft_frequencies[1] - 1/N ) <= 0.05), f'fft frequncies incorrect:  {fft_frequencies[1]} does not equal {1/N}'
assert(abs(fft_frequencies[N-1] + 1/N ) <= 0.05), f'fft frequncies incorrect:  {fft_frequencies[N-1]} does not equal {-1/N}'

for (v1, v2, v3, v4) in zip(weekly_prices, upto_1_year,one_year_to_1_quarter,less_than_1_quarter ):
    assert ( abs(v1 - (v2 + v3+v4)) <= 0.01), 'The components are not adding up -- there is a mistake in the way you split your original signal into various components'
print('All tests OK -- 10 points!!')