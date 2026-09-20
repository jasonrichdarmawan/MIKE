# %%

from statistics import NormalDist

# %%

z = NormalDist().inv_cdf(0.975)  # 95% confidence interval

for name, x, n in [
    ('All sampled outputs',23,440)
    ('MQuAKE overall',14,320),
    ('MQuAKE English',9,160),
    ('MQuAKE Chinese',5,160),
    ('Two-Hop',9,120),
]:
    p = x/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
    print(f'{name}: {100*p:.2f}% ({100*(c-h):.2f}%, {100*(c+h):.2f}%)')

# %%