      SUBROUTINE SMOr3p(N,f1,f2)

c 3/point running average for regular gridded data.
c 
c
    
      INTEGER I,N
      REAL f1(N), f2(N), f(1000)


c copy input array f1 to local storage array (maxlength =  1000 point)

      DO I = 1, N

         f(i) = f1(i)

      END DO

c Do the 3-point running average and copy to output array f2     

      f2(1) = f(1)
      f2(N) = f(N)

      DO I = 2, N-1

        f2(i) = 0.25* (f(I-1) + 2*f(i) + f(I+1))

      END DO

C return

      END SUBROUTINE



