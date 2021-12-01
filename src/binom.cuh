#ifndef binom_INCLUDED
#define binom_INCLUDED

#define CUPY_USE_BINOMIAL


__device__ unsigned long rk_random(rk_state *state) {
    unsigned int *xor128 = state->xor128;
    unsigned int t = xor128[0] ^ (xor128[0] << 11);
    xor128[0] = xor128[1];
    xor128[1] = xor128[2];
    xor128[2] = xor128[3];
    return xor128[3] ^= (xor128[3] >> 19) ^ t ^ (t >> 8);
}
__device__ double rk_double(rk_state *state) {
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    int a = rk_random(state) >> 5, b = rk_random(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

__device__ long rk_binomial_btpe(rk_state *state, long n, double p) {
    double r,q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4;
    double a,u,v,s,F,rho,t,A,nrq,x1,x2,f1,f2,z,z2,w,w2,x;
    long m,y,k,i;
    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p)) {
        /* initialize */
        state->nsave = n;
        state->psave = p;
        state->has_binomial = 1;
        state->r = r = min(p, 1.0-p);
        state->q = q = 1.0 - r;
        state->fm = fm = n*r+r;
        state->m = m = (long)floor(state->fm);
        state->p1 = p1 = floor(2.195*sqrt(n*r*q)-4.6*q) + 0.5;
        state->xm = xm = m + 0.5;
        state->xl = xl = xm - p1;
        state->xr = xr = xm + p1;
        state->c = c = 0.134 + 20.5/(15.3 + m);
        a = (fm - xl)/(fm-xl*r);
        state->laml = laml = a*(1.0 + a/2.0);
        a = (xr - fm)/(xr*q);
        state->lamr = lamr = a*(1.0 + a/2.0);
        state->p2 = p2 = p1*(1.0 + 2.0*c);
        state->p3 = p3 = p2 + c/laml;
        state->p4 = p4 = p3 + c/lamr;
    } else {
        r = state->r;
        q = state->q;
        fm = state->fm;
        m = state->m;
        p1 = state->p1;
        xm = state->xm;
        xl = state->xl;
        xr = state->xr;
        c = state->c;
        laml = state->laml;
        lamr = state->lamr;
        p2 = state->p2;
        p3 = state->p3;
        p4 = state->p4;
    }
  /* sigh ... */
  Step10:
    nrq = n*r*q;
    u = rk_double(state)*p4;
    v = rk_double(state);
    if (u > p1) goto Step20;
    y = (long)floor(xm - p1*v + u);
    goto Step60;
  Step20:
    if (u > p2) goto Step30;
    x = xl + (u - p1)/c;
    v = v*c + 1.0 - fabs(m - x + 0.5)/p1;
    if (v > 1.0) goto Step10;
    y = (long)floor(x);
    goto Step50;
  Step30:
    if (u > p3) goto Step40;
    y = (long)floor(xl + log(v)/laml);
    if (y < 0) goto Step10;
    v = v*(u-p2)*laml;
    goto Step50;
  Step40:
    y = (long)floor(xr - log(v)/lamr);
    if (y > n) goto Step10;
    v = v*(u-p3)*lamr;
  Step50:
    k = labs(y - m);
    if ((k > 20) && (k < ((nrq)/2.0 - 1))) goto Step52;
    s = r/q;
    a = s*(n+1);
    F = 1.0;
    if (m < y) {
        for (i=m+1; i<=y; i++) {
            F *= (a/i - s);
        }
    } else if (m > y) {
        for (i=y+1; i<=m; i++) {
            F /= (a/i - s);
        }
    }
    if (v > F) goto Step10;
    goto Step60;
    Step52:
    rho = (k/(nrq))*((k*(k/3.0 + 0.625) + 0.16666666666666666)/nrq + 0.5);
    t = -k*k/(2*nrq);
    A = log(v);
    if (A < (t - rho)) goto Step60;
    if (A > (t + rho)) goto Step10;
    x1 = y+1;
    f1 = m+1;
    z = n+1-m;
    w = n-y+1;
    x2 = x1*x1;
    f2 = f1*f1;
    z2 = z*z;
    w2 = w*w;
    if (A > (xm*log(f1/x1)
           + (n-m+0.5)*log(z/w)
           + (y-m)*log(w*r/(x1*q))
           + (13680.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320.
           + (13680.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z/166320.
           + (13680.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320.
           + (13680.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.)) {
        goto Step10;
    }
  Step60:
    if (p > 0.5) {
        y = n - y;
    }
    return y;
}
__device__ long rk_binomial_inversion(rk_state *state, int n, double p) {
    double q, qn, np, px, U;
    long X, bound;
    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p)) {
        state->nsave = n;
        state->psave = p;
        state->has_binomial = 1;
        state->q = q = 1.0 - p;
        state->r = qn = exp(n * log(q));
        state->c = np = n*p;
        state->m = bound = min((double)n, np + 10.0*sqrt(np*q + 1));
    } else {
        q = state->q;
        qn = state->r;
        np = state->c;
        bound = state->m;
    }
    X = 0;
    px = qn;
    U = rk_double(state);
    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = rk_double(state);
        } else {
            U -= px;
            px  = ((n-X+1) * p * px)/(X*q);
        }
    }
    return X;
}

//rk_binomial
__device__ long rk_binomial(rk_state *state, int n, double p) {
    double q;
    if (p <= 0.5) {
        if (p*n <= 30.0) {
            return rk_binomial_inversion(state, n, p);
        } else {
            return rk_binomial_btpe(state, n, p);
        }
    } else {
        q = 1.0-p;
        if (q*n <= 30.0) {
            return n - rk_binomial_inversion(state, n, q);
        } else {
            return n - rk_binomial_btpe(state, n, q);
        }
    }
}

__device__ void rk_seed(unsigned long long s, rk_state *state) {
    for (int i = 1; i <= 4; i++) {
        s = 1812433253U * (s ^ (s >> 30)) + i;
        state->xor128[i - 1] = s;
    }
    state->has_gauss = 0;
    state->has_binomial = 0;
}


__device__ void csd_multinomial (rk_state * r, int K, int N, double *p, long *n)
{
  int k;
  double norm = 0.0;
  double sum_p = 0.0;

  long sum_n = 0;

  /* p[k] may contain non-negative weights that do not sum to 1.0.
   * Even a probability distribution will not exactly sum to 1.0
   * due to rounding errors. 
   */

  for (k = 0; k < K; k++){
    norm += p[k];
  }

  for (k = 0; k < K; k++){
    if (p[k] > 0.0)
    {
      n[k] = rk_binomial (r,  N - sum_n, p[k] / (norm - sum_p));
    }
    else
    {
      n[k] = 0;
    }

    sum_p += p[k];
    sum_n += n[k];
  }

}

__device__ void csd_int_multinomial (rk_state * r, int K, int N, float *p, int *n)
{
  int k;
  float norm = 0.0;
  float sum_p = 0.0;

  int sum_n = 0;

  /* p[k] may contain non-negative weights that do not sum to 1.0.
   * Even a probability distribution will not exactly sum to 1.0
   * due to rounding errors. 
   */

  for (k = 0; k < K; k++){
    norm += p[k];
  }

  for (k = 0; k < K; k++){
    if (p[k] > 0.0)
    {
      n[k] = (int)rk_binomial (r,  N - sum_n, p[k] / (norm - sum_p));
    }
    else
    {
      n[k] = 0;
    }

    sum_p += p[k];
    sum_n += n[k];
  }

}

#endif