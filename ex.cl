float
alpha(int j)
{
  if( 1 <= j && j <= (64-1))
    return 1;
  else
    return 0;
}


float
beta (int j)
{
  return 0;
}

__kernel void
proc (__global const float *t_g,
      __global float       *res_g)
{
  int gid = get_global_id (0);

  int N = 64;
  int c = 1;
  int m = 1;
  int t = t_g[gid];
  int j = 0;
  float wk;
  float wn = 2 * sqrt ((float) c / (float) m);
  float tmp;

  tmp = 0;
  {
    for (int i = 1; i <= N; i++)
      tmp += alpha(i);
  }
  res_g[gid] = tmp / N;

  tmp = 0;
  {
    for (int i = 1; i <= N; i++)
      tmp += beta(i);
  }
  res_g[gid] += tmp * t / N;

  tmp = 0;
  {
    for (int k = 1; k <= N / 2 - 1; k++) {
      wk = 2 * sqrt ((float) c / (float) m) * sin (M_PI_F * k / N);
      float tmp1, tmp2;

      tmp1 = 0;
      {
        tmp2 = 0;
        for (int i = 1; i <= N; i++)
          tmp2 += beta(i) * cos (2 * M_PI_F * k * i / N);
        tmp1 += 2 * tmp2 * sin (wk * t) / (wk * N);

        tmp2 = 0;
        for (int i = 1; i <= N; i++)
          tmp2 += alpha(i) * cos (2 * M_PI_F * k * i / N);
        tmp1 += 2 * tmp2 * cos (wk * t) / N;

        tmp1 *= cos (2 * M_PI_F * k * j / N);
      }
      tmp += tmp1;

      tmp1 = 0;
      {
        tmp2 = 0;
        for (int i = 1; i <= N; i++)
          tmp2 += beta(i) * sin (2 * M_PI_F * k * i / N);
        tmp1 += 2 * tmp2 * sin (wk * t) / (wk * N);

        tmp2 = 0;
        for (int i = 1; i <= N; i++)
          tmp2 += alpha(i) * sin (2 * M_PI_F * k * i / N);
        tmp1 += 2 * tmp2 * cos (wk * t) / N;

        tmp1 *= sin (2 * M_PI_F * k * j / N);
      }
      tmp += tmp1;
    }
  }
  res_g[gid] += tmp;

  tmp = 0;
  {
    float tmp1;

    tmp1 = 0;
    for (int i = 1; i <= N; i++)
      tmp1 += pow (-1, (float) i) * beta(i);
    tmp += tmp1 * sin (wn * t) / (wn * N);

    tmp1 = 0;
    for (int i = 1; i <= N; i++)
      tmp1 += pow (-1, (float) i) * alpha(i);
    tmp += tmp1 * cos (wn * t) / N;
  }
  res_g[gid] += tmp * pow (-1, (float) j);
}
