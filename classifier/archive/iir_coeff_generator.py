from scipy.signal import iirfilter

b, a = iirfilter(
    N=4,
    Wn=[0.01, 0.99],  # determine f_c1 and f_c2
    btype='bandpass',
    ftype='butter',
    output='ba'
)

print(f'a coeffs: {a}')
print(f'b coeffes: {b}')

'''
with Wn = [0.01. 0.99]
a coeffs: [ 1.00000000e+00 -2.44249065e-15 -3.83582554e+00  5.77315973e-15
  5.52081914e+00 -1.02140518e-14 -3.53353522e+00  1.99840144e-15
  8.48555999e-01]
b coeffes: [ 0.92117099  0.         -3.68468397  0.          5.52702596  0.
 -3.68468397  0.          0.92117099]
'''
