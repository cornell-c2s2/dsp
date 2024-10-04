from scipy.signal import iirfilter

b, a = iirfilter(
        N=4,
        Wn=[f_c1,f_c2], # determine f_c1 and f_c2
        btype='bandpass',
        ftype='butter',
        output='ba'
        )

print(f'a coeffs: {a}')
print(f'b coeffes: {b}')