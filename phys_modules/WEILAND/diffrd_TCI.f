      SUBROUTINE DIFFRD_TCI(ZRP, IR, FT, U, CHI, CHE, D, CHQ, DQ, FL, 
     &                      GRKVOT, GSHEAR,KFORM) 
C----------------------------------------------------------------------- 
C THIS SUBROUTINE CALCULATES THE TRANSPORT MATRIX FOR THE WEILAND 
C TRANSPORT MODEL.  
C-----------------------------------------------------------Q------------ 
C Modified: P. STRAND, ORNL, 2000-11-10                strandpi@ornl.gov  
C----------------------------------------------------------------------- 
C THE LINEAR DISPERSION SOLVER IS ASSUMED TO HAVE BEEN CALLED BEFORE  
C THIS SUBROUTINE CALL AND THE IR UNSTABLE ROOTS ARE PASSED IN THROUGH  
C THE COMPLEX RP ARRAY.  
C THE INPUT AND OUTPUT TAKES SLIGHTLY DIFFERENT FORM DEPENDING ON THE  
C INTEGER KFORM FLAG. 
C  
C  . THE ORIGINAL (CHALMERS FORM) OF THE TRANSPORT MATRIX IS RETURNED  
C    FOR KFORM = 0, THE INPUT U VECTOR IS HERE ASSUMED TO BE ORDERED 
C    U=(TI,TE,N,TQ,NQ) WHERE N IS THE ELECTRON DENSITY.  
C 
C  . A MODIFIED VERSION IS RETURNED FOR ALL OTHER VALUES: 
C     - A FACTOR 3/2 IS ADDED TO THE HEAT DIFFUSIVITES AND CORRESPONDING 
C       PINCH TERMS (FL) 
C     - IT IS EXPRESSED IN TERMS OF A MAIN ION DENSITY 
C     - IT IS DIMENSIONALLY A DIFFUSION MATRIX [M^2/S], CONSEQUENTLY 
C       TI/NI AND SIMILAR TERMS ARE TO BE ADDED IN APPROPRIATELY TO 
C       GIVE  AN ENERGY FLUX. 
C    THE INPUT U VECTOR IS HERE ASSUMED TO BE ORDERED 
C    U=(TI,TE,N,TQ,NQ) WHERE N IS THE ION  DENSITY. <---- NOTE!!! 
C------------------------------------------------------------------------ 
C IN THIS VERSION ELECTROMAGNETIC CORRECTIONS HAVE BEEN ADDED FOR THE  
C CASE WHERE THE LINEAR DISPERSION RELATION IS OF ORDER 9 OR HIGHER. 
C BECAUSE IF THIS THE SHEAR FLOW REDUCTION OF THE LINEAR GROWTH RATES 
C HAVE TO BE INCLUDED DIRECTLY IN THE LINEAR DISPERSION SOLVER BEFORE  
C THIS ROUTINE IS CALLED. 
C J. Weiland,
C-----------------------------------------------------------------------  
      IMPLICIT NONE 

      INTEGER, PARAMETER :: RP=KIND(1.0D0) 
C 
C ON INPUT 
C 
      COMPLEX(RP) ZRP(11)       ! THE SOLUTIONS FROM THE DISPERSION EQ. 
      INTEGER    IR           ! NUMBER OF NONZERO SOLUTIONS 
      REAL(RP)     FT           ! FRACTION OF TRAPPED ELECTRONS 
      REAL(RP)     U(5)         ! U = (TI, TE, N, TQ, NQ) 
      REAL(RP)     GRKVOT       ! FLUX SURFACE CORRECTION 
      REAL(RP)     GSHEAR       ! NORMALIZED SHEARING RATE 
      INTEGER    KFORM        ! MATRIX FORM (CHALMERS -0)  
                              ! (JETTO VERSION OTHERWISE)  
C 
C ON OUTPUT 
C 
      REAL(RP)     CHI(5)       ! FIRST  ROW IN DIFFUSION MATRIX 
      REAL(RP)     CHE(5)       ! SECOND ROW IN DIFFUSION MATRIX 
      REAL(RP)     D  (5)       ! THIRD  ROW IN DIFFUSION MATRIX 
      REAL(RP)     CHQ(5)       ! FOURTH ROW IN DIFFUSION MATRIX 
      REAL(RP)     DQ (5)       ! FIFTH  ROW IN DIFFUSION MATRIX 
      REAL(RP)     FL (5)       ! CORRESPONDING CONVECTIVE PINCH TERMS 
C 
C COMMON BLOCKS  --  A lot of redundancy here 
C 
      REAL(RP)  THRD, FTR, FTRT,  STR, RFL, D1, XIH 
      INTEGER IN, IW 
      COMMON  /PAR/    THRD, FTR, STR, RFL, D1, XIH, IN, IW 
C 
      REAL(RP)  SCHI, SCHE, SD, SCHQ, SDQ 
      COMMON /EFFDIFF/  SCHI, SCHE, SD, SCHQ, SDQ 
C 
      REAL(RP)  EN, EI, EE 
      COMMON /GRAD/ EN,EI,EE 
      REAL(RP)  ENH             !Not really needed since we are using 
      COMMON /GRADADD/ ENH    !scale lengths directly instead 
C 
      REAL(RP) BQ, G, SI, Z, TAUZ, ENQ, EQ, KIQ, KXQ 
      COMMON /WIMP/ BQ, G, SI, Z, TAUZ, ENQ, EQ, KIQ, KXQ 
c 
      REAL(RP) BTA,BT1,VEF 
      COMMON /COLL/ BTA, VEF 
c 
      COMPLEX(RP) ZZ(11) ! All eigenvalues needed 
      COMMON/ZZ/ ZZ 
c 

      REAL(RP) EM
      COMMON/EM/EM

      REAL(RP)  ZVR(11,11), ZVI(11,11) ! Eigenvectors 
      COMMON/ZV/ ZVR,ZVI 
c 
      INTEGER NEQ 
      COMMON/NEQ/ NEQ 
c 
      COMPLEX(RP) IU

      REAL(RP)  ALP, KPS ! Temporary fix
      COMMON/TEST/ALP
      COMMON /MTEST/ KPS
      
      REAL(RP) betae, q, S, CS
      COMMON/BETAE/ betae, q, S, Cs     

      REAL(RP) VTOR, LVFT, BP2BT, DMTEF
      COMMON /WMOMTRAN/VTOR, LVFT, BP2BT, DMTEF

      REAL(RP) VPOL, LVFP
      COMMON /WMOMTRANXXX/ VPOL, LVFP

      REAL(RP) XND, NJR, NJI, NJ2, SDI
      REAL(RP) XDDIA(6), XDPINCH(6)
      COMMON /XDOUT/ XDDIA, XDPINCH
      COMMON /EFFDIFFXXX/ SDI
      
      REAL(RP) GAV0, RAV0
      COMMON /GAVROT/ GAV0, RAV0

      REAL(RP) dma, dmb, dmc
      COMMON /DMROT/dma, dmb, dmc

C----------------------------------------------------------------------- 
C LOCAL VARIABLES 
C----------------------------------------------------------------------- 
  
      COMPLEX(RP) W                ! LOCAL COPY OF RP 
      COMPLEX(RP) NE               ! DENOMINATOR IN EXPRESSIONS 
C 
      REAL(RP)     WR               ! REAL PART OF W 
      REAL(RP)     WI               ! IMAGINARY PART OF W 
C 
      REAL(RP)     XI(5), RCI, DNI       ! PART OF CHI 
      REAL(RP)     XE(5), XEH, RCE, DNE  ! PART OF CHE 
      REAL(RP)     XD(5), XDH, RD        ! PART OF D 
      REAL(RP)     XDQ(5), RNQ           ! PART OF Dq 
      REAL(RP)     XQ(5), XQH            ! PART OF CHQ 
C 
      REAL(RP)     WSQ    ! 
      REAL(RP)     NN,NR,NI 
      REAL(RP)     TAUI         ! TI/TE 
      REAL(RP)     TVR, N , TE,TI, pi,pe, pn, HP, pnq, piq 
C 
      INTEGER    I,J, IJ 
 
C ALL SCALELENGTHS SHOULD BE DIMENSIONLESS (OR HAVE THE DIMENSION OF 
C I.E. LTI=-TI/dDTI/dRHO, WITH THE eEXCEPTION OF LB = R  
 
      REAL(RP) Lti, lte, ln, Lnq, Lb , ltq, Lni 
      COMMON /LSCALES/ LTI, LTE, LN, LNQ, LB , LTQ, LNI 
  
c 
c... impurities 
c 
      REAL(RP) eni,  ng, tz, nq, tq, dnq, imp, nimp, nqr 
      REAL(RP) nqi, k, ts, gi, rcq, dqt, h1,h2,pq 
      REAL(RP) a2,a3, b, c1,c2, dh,e1,e2,F,dqn1, dqn2,t1,t2 
      REAL(RP) kpc, R0 
c ... K-parallel terms
      REAL(RP) n1r, ndz1, ndz2, ndz3, dqn3 
c 
c... Collisions 
c 
      COMPLEX(RP) GA, GB, GM, BT2 
c 
      REAL(RP) GAR, GAI, GBR, GBI, HR,XDE,XDI 
      REAL(RP) DIVGA1, DIVGA2, DIVGA3 
      REAL(RP) DIVA1, DIVA2, DIVA3 
      REAL(RP) DIVGB, DIVB 
      REAL(RP) DEVGA1, DEVGA2, DEVGA3 
      REAL(RP) DEVA1, DEVA2, DEVA3 
      REAL(RP) DEVGB, DEVB 
      REAL(RP) SVB,PHS, WII 
      REAL(RP) DT,SHPE,SCHEF,DEF,ELNR,ELNI,AINF,HPE,IMF,CEFT 
      REAL(RP) DEFT,GEI,ETE, OM_DE 
      COMPLEX(RP) FIH,NEF,TEF,AV,NRAT,FRAT,ELN  !! AV added! 
c 
c --- New collision variables included in the modified model -- 
c 
      REAL(RP) RVAID,RVAC,RVAC1,RVAC2,RVAC3 
      REAL(RP) RVAID1,RVAID2,RVAID3
      REAL(RP) SVAID,SVAC,SVAC1,SVAC2,SVAC3 
      REAL(RP) SVAID1,SVAID2,SVAID3 
      REAL(RP) NIER,NIEI,RN,YDA,RVB 
      COMPLEX(RP) NIE 
      
c
c --- Momentum transport
c

      REAL(RP) Kkap,KAP1,Hm, KS
      COMPLEX (RP) Fm,hf, KPF
     
      COMPLEX(RP) TII, NIF, AVRAT, MRT, DMSP, ELMS   
      REAL(RP) DMS,DMI, DMST, DMIT, D_MD, DMIP, GP1, GP2   
      
C----------------------------------------------------------------------- 


      dma = 0.0_RP
      dmb = 0.0_RP
      dmc = 0.0_RP

C----------------------------------------------------------------------- 
C  LOCAL COPIES OF THE FUNCTION VALUES --- moved pis may03
C----------------------------------------------------------------------- 
  
      NQ=U(5) 
      TQ=U(4) 
      N=U(3) 
      TE=U(2) 
      TI=U(1) 
      TAUI=TI/TE 
  
C      
       TVR=2.0_RP/3.0_RP 
       FTRT=FTR*TAUI
       kpc=1.0_RP       !! kpc = 1 due to normalisation of AV 
       IU=(0.0_RP,1._RP)
c... Impurities 
      G=1.0_RP-Z*BQ        
      NG=MAX(G,1.e-10_RP) 
      GI=1.0_RP/NG 
C 
      TZ=Z*TAUZ 
      IF(abs(TZ).GT.0.0001_RP) IMP = max(1.0_RP/TZ, 0.0001_RP) 
      ENI=1.0_RP/EN

C----------------------------------------------------------------------- 
C SET DIFFUSION MATRIX AND EFFECTIVE DIFFUSIVITIES INITIALLY TO ZERO 
C----------------------------------------------------------------------- 
  
      SCHI=0.0_RP
      SCHE=0.0_RP
      SD=0.0_RP
      SCHQ=0.0_RP 
      SDQ=0.0_RP
C
      XND = 0.0D0
C 
      DO I = 1, 5 
         CHI(I)=0.0_RP 
         CHE(I)=0.0_RP
         D(I)=0.0_RP 
         CHQ(I)=0.0_RP 
         DQ(I)=0.0_RP 
         FL(I)=0.0_RP 
      END DO 
  
      DO I = 1, 6
        XDDIA(I) = 0.0_RP
        XDPINCH(I)= 0.0_RP
      END DO

C 
      PI  = 0.0_RP
      PIQ = 0.0_RP 
      HP  = 0.0_RP
      PE  = 0.0_RP
      PN  = 0.0_RP
      PQ  = 0.0_RP
      PNQ = 0.0_RP 
      RNQ = 0.0_RP
      RCQ = 0.0_RP
C 
      XIH=0.0_RP 
C 
      IF (IR.EQ.0) RETURN   ! No unstable modes ! 
  
      R0= LB 
      OM_DE = 2.0_RP*RFL/LB*0.311e6_RP*SQRT(TE/2.0_RP)    
      DO I=1,IR 
         W=ZRP(I) 
         WR=DREAL(W) 
         WI=DIMAG(W)-ABS(GSHEAR)
         W=DCMPLX(WR,WI) 
         if (WI.lt.0.0_RP) GOTO 2000  
         GM=1.d0+LB/2.0_RP/LTE/(W-1.0_RP+(0.0_RP,1.0_RP)*vef) !EE/ENI = LB/2*lte 
         BT1=BTA-2.5_RP 
         BT2=BTA-2.5_RP*GM 
         WSQ=WR*WR + WI*WI 
         WII=1.0_RP/WI 
C 
         NE=W*W - 2.0_RP*FTR*W+FTR+(0.0_RP,1.0_RP)*VEF*(W-FTR+TVR*BT1)
         NR=DREAL(NE) 
         NI=DIMAG(NE) 
c 
         NN=NR*NR + NI*NI 
C 
         DNI=(WR+FTR*TAUI*GAV0)*(WR+FTR*TAUI*GAV0)+WI*WI 
         DNE=(WR-FTR)*(WR-FTR)+WI*WI 
         DNQ=(WR+FTR*IMP*GAV0)**2+WI*WI 
  
         XDH=D1*sqrt(Te)**3*WI**3/RFL 
         XIH=XDH/DNI 
         XEH=XDH/DNE 
         XQH=XDH/DNQ 
  
c------------------------------------------------------------------------ 
c Collisions... 
c------------------------------------------------------------------------ 
      IF(VEF.GT.0.0_RP) THEN
C
C   ******   COLLISION PARAMETERS  *******
C
      GA=W-FTR+TVR*BT2
      GB=(W-FTR)/(W-1.D0+IU*VEF)
      GAR=DREAL(GA)
      GAI=DIMAG(GA)
      GBR=DREAL(GB)
      GBI=DIMAG(GB)
      HR=WR-FTR+TVR*BT1
      XDE=WSQ-FTR*WR
      XDI=WSQ+FTR*TAUI*WR
      YDA=WR*(1.0_RP-EN)+EE-STR+FTR*EN
      NIE=W*W-2.0_RP*FTR*W+FTR 
      NIER=DREAL(NIE) 
      NIEI=DIMAG(NIE) 
C Linear trapped electron density response dn?n = FX ephi/Te where **
C **  FX = ENI*(YDA+IU*WI*(1.0_RP-EN)+IU*VEF*(EN*GA+EE*GB))/NE **
C ** where NE=NER+IU*NEI; NER=NIER-WI*VEF, NEI=NIEI+VEF*HR ****
C   ***************************************
C
C    We write FX = KK*(KR + IU*KI) where KK=1/(NN*EN),
C    KR = RVA + EE*RVB and KI = SVA + EE*SVB
C    We divide into ideal and collisional parts as:
C    RVA = RVAID + RVAC, SVA = SVAID + SVAC 
C
      RVAID=NIER*YDA+NIEI*WI*(1.0_RP-EN)
      SVAID=NIER*WI*(1.0_RP-EN)-NIEI*YDA
c
      RVAC=VEF*(EN*(NI*GAR-NR*GAI)-WI*YDA+HR*WI*(1.0_RP-EN))
      RVB=VEF*(NI*GBR-NR*GBI)
c
      SVAC=VEF*(EN*(NR*GAR+NI*GAI)-(1.0_RP-EN)*WI**2-HR*YDA)
      SVB=VEF*(NR*GBR+NI*GBI)
c
c     These parts are now divided into diagonal, off diagonal and convective 
c     parts as e.g.  RVA = RVA1 + EN*RVA2 + EE*RVA3 etc ...
c
c ** The following parts, due to the ideal part of the density response
c    enter only for electron thermal transport -----------------------
c
      RVAID1=NIER*(WR-STR)+WI*NIEI
      RVAID2=NIER*(FTR-WR)-WI*NIEI
      RVAID3=NIER
c
      SVAID1=WI*NIER-NIEI*(WR-STR)
      SVAID2=-WI*NIER+NIEI*(WR-FTR)
      SVAID3=-NIEI
c----------------------------------------------------------------------
c
      RVAC1=VEF*WI*(HR-WR+STR)
      RVAC2=VEF*(GAR*NI-GAI*NR+WI*(WR-FTR-HR))
      RVAC3=-VEF*WI
c
      SVAC1=VEF*(HR*(STR-WR)-WI*WI)
      SVAC2=VEF*(GAR*NR+GAI*NI+WI*WI+HR*(WR-FTR))
      SVAC3=-VEF*HR
c
c     The ideal parts RVAID etc .. will only enter into the electron thermal 
c     transport. For the particle transport we need only SVAC and SVB
c
c  Ion thermal conductivity
c
      DIVGA1=FTRT*RVAC1
      DIVGA2=FTRT*RVAC2
      DIVGA3=FTRT*RVAC3
c
      DIVA1=XDI*SVAC1
      DIVA2=XDI*SVAC2
      DIVA3=XDI*SVAC3
c
      DIVGB=FTRT*RVB
      DIVB=XDI*SVB
c
c   Electron thermal conductivity
c
      DEVGA1=-FTR*RVAC1-BT1*VEF*(SVAID1+SVAC1)
      DEVGA2=-FTR*RVAC2-BT1*VEF*(SVAID2+SVAC2)
      DEVGA3=-FTR*RVAC3-BT1*VEF*(SVAID3+SVAC3)
c
      DEVA1=XDE*SVAC1-BT1*VEF*(WR-FTR)*(RVAID1+RVAC1)
      DEVA2=XDE*SVAC2-BT1*VEF*(WR-FTR)*(RVAID2+RVAC2)
      DEVA3=XDE*SVAC3-BT1*VEF*(WR-FTR)*(RVAID3+RVAC3)
c
      DEVGB=-(FTR*RVB+BT1*VEF*SVB)
      DEVB=XDE*SVB-BT1*VEF*(WR-FTR)*RVB
c
      RN=1.0_RP/NN

      DIVGA1=DIVGA1*RN
      DIVGA2=DIVGA2*RN
      DIVGA3=DIVGA3*RN
      DIVGB=DIVGB*RN
      DEVGA1=DEVGA1*RN
      DEVGA2=DEVGA2*RN
      DEVGA3=DEVGA3*RN
      DEVGB=DEVGB*RN
      DIVA1=DIVA1*RN
      DIVA2=DIVA2*RN
      DIVA3=DIVA3*RN
      DIVB=DIVB*RN
      DEVA1=DEVA1*RN
      DEVA2=DEVA2*RN
      DEVA3=DEVA3*RN
      DEVB=DEVB*RN
      SVAC1=SVAC1*RN
      SVAC2=SVAC2*RN
      SVAC3=SVAC3*RN
      SVB=SVB*RN
C
      PHS  = (WR-FTR)*DREAL(BT2)+WI*DIMAG(BT2) 
c
         ELSE  
c... Zero out the collisionality contributions if not wanted 
            DIVGA1=0.0_RP 
            DIVGA2=0.0_RP
            DIVGA3=0.0_RP
            DIVGB=0.0_RP 
 
            DIVA1=0.0_RP
            DIVA2=0.0_RP
            DIVA3=0.0_RP 
            DIVB=0.0_RP 
 
            DEVGA1=0.0_RP 
            DEVGA2=0.0_RP
            DEVGA3=0.0_RP
 
            DEVGB=0.0_RP 
            DEVA1=0.0_RP
            DEVA2=0.0_RP
            DEVA3=0.0_RP
 
            DEVB=0.0_RP 
 
            SVAC1=0.0_RP 
            SVAC2=0.0_RP
            SVAC3=0.0_RP
            SVB=0.0_RP 
            PHS=0.0_RP
            RN=0.0_RP

         END IF 
  
c------------------------------------------------------------------------ 
C Contributions to the ion conductivity 
c------------------------------------------------------------------------ 
         B =(WSQ*(2.0_RP*(WR-FTR)+FTR*TAUI) - FTR*FTR*TAUI)/NN 
 
C... Necessary splitting:  A = A2*EN + A3 
         A2=(WSQ*WSQ+FTR*((STR+FTR*TAUI-2.0_RP*WR)*WSQ+FTR*TAUI*(FTR- 
     &        2.0_RP*WR)))/NN 
         A3=(WSQ*(-WSQ+2.0_RP*STR*WR-FTR*(11.0_RP/3.0_RP+TAUI))
     &         +FTR*FTR*TAUI*(2.0_RP*WR-STR))/NN 
 
c------------------------------------------------------------------------ 
C Contributions to the electron conductivity 
c------------------------------------------------------------------------ 
          DH =(WSQ*(2.0_RP*WR-5.0_RP)+FTR*FTR)/NN 
  
c... Necessary splitting: 1+C = C1 + C2*EN 
         C1 = (NN - WSQ*WSQ+14.0_RP/3.0_RP*WSQ*WR-40.0_RP/9.0_RP*WSQ 
     &        - 50.0_RP/9.0_RP*WR + 175.0_RP/27.0_RP) / NN 
         C2 = (WSQ*WSQ-10.0_RP/3.0_RP*WSQ*WR + 10.0_RP/9.0_RP*WSQ 
     &        + 50.0_RP/9.0_RP*WR - 125.0_RP/27.0_RP) / NN 
  
c------------------------------------------------------------------------ 
C   Contributions to the electron diffusivity 
c------------------------------------------------------------------------ 
         F  = 2.0_RP*(-WR+FTR)/NN 
 
c... Necessary splitting E = E1 + E2*EN 
         E1 = (WSQ+11.0_RP*THRD*FTR-2.0_RP*WR*STR)/NN 
         E2 = -(-2.0_RP*WR*FTR + (WSQ + STR*FTR))/NN 
  
c------------------------------------------------------------------------ 
c Impurities 
c------------------------------------------------------------------------  
         NQR=WR**2-WI**2+2.0_RP*FTR*IMP*WR+FTR*IMP*IMP 
         NQI=2.0_RP*WI*(WR+FTR*IMP) 
         NIMP=NQR**2+NQI**2 
         N1R=WR-2.0_RP*IMP
         NDZ1=Z/(2.0_RP*Z*4.0_RP*q**2*RFL*RFL*(N1R**2+WI**2))
         NDZ2=NQR*11.0_RP*IMP/3.0_RP
         NDZ3=2.0_RP*(WR+FTR*IMP)*(WI**2+N1R*(WR+FTR*IMP))
  
c------------------------------------------------------------------------ 
C Contributions to the main ion Conductivity 
c------------------------------------------------------------------------ 
         K=IMP*(FTR*FTR*TAUI*IMP*IMP-WSQ*(2.0_RP*WR 
     &     +FTR*(2._RP*IMP+TAUI)))/NIMP 
  
c... Necessary Splitting: H = H1 + H2*enq 
         H1=(WSQ*(-WSQ-2._RP*IMP*WR*STR 
     &     +FTR*IMP*IMP*(-11.0_RP*THRD)+FTR*TAUI*IMP) 
     &     +FTR*FTR*TAUI*IMP*IMP*(2.0_RP*WR+STR*IMP))/NIMP 
         H2=(WSQ*
     &      (WSQ+2.0_RP*IMP*WR*FTR+FTR*IMP*IMP*STR-FTR*TAUI*IMP*FTR) 
     &     -FTR*FTR*TAUI*IMP*IMP*(2.0_RP*WR-FTR*IMP))/NIMP 

c------------------------------------------------------------------------ 
c Simple estimate for the diagonal contribution of the main ion diffusivity
c------------------------------------------------------------------------ 
         NJR = WR**2 - WI**2 + 2.0_RP*FTRT*WR+FTRT**2/FTR
         NJI = 2.0_RP*WI*(wr+FTRT)
         NJ2 = NJR**2 + NJI**2
  
c------------------------------------------------------------------------ 
c Contributions to the impurity conductivity 
c------------------------------------------------------------------------ 
        TS = IMP*(FTR*FTR*IMP**3-WSQ*(2.0_RP*WR+5.0_RP*IMP))/NIMP*KXQ 
 
c... necessary Splitting: T = T1 + T2*ENQ 
         T1 = (Wsq*(-WSQ-2.0_RP*IMP*WR*STR-FTR*IMP*IMP*8.0_RP*THRD) 
     &      +FTR*FTR*IMP**3*(2.0_RP*WR+IMP*STR))/NIMP*KXq 
         T2 = (WSQ*(WSQ+2.0_RP*IMP*WR*FTR+FTR*IMP*IMP*TVR) 
     &      -FTR*FTR*IMP**3*(2.0_RP*WR+IMP*FTR))/NIMP*KXq 
  
  
c------------------------------------------------------------------------ 
c Contributions to the impurity diffusivity 
c------------------------------------------------------------------------ 
         DQT=2.D0*IMP*(WR+FTR*IMP)/NIMP 
  
c... Necessary splitting: DQN = DQN1 + DQN2*EN 
         DQN1 = (-NQR+2.0_RP*(WR+IMP*STR)*(WR+FTR*IMP))/NIMP 
         DQN2 =  (NQR-2.0_RP*(WR+IMP*FTR)*(WR+FTR*IMP))/NIMP 
         DQN3 = -NDZ1*(NDZ2+NDZ3)/NIMP

C----------------------------------------------------------------------- 
C Ion Row of the transport matrix  
C----------------------------------------------------------------------- 
c        --- Chalmers implementation  
         IF (kform .eq. 0) THEN 
            XI(1) = XIH 
            XI(2) = TVR*Ft*XIH*TAUI*GI*(B-DIVGA3-DIVGB-
     &              (DIVA3+DIVB)*WII) 
            XI(3) = -TVR*XIH*TE/N*TAUI*(Ln/Lni+ 
     &               ft*GI*(A3+DIVGA1+DIVA1*WII)) 
            IF(NQ .GT. 1.0e-7_RP) THEN 
               XI(4) = -XIH*TVR*BQ*Z*GI*K*TAUZ*TAUI 
               XI(5) = TVR*XIH*Ti/N*Z*H1*GI 
            ENDIF 
 
c... Convective part of the ion heat flux  
            HP  = XIH*TVR*TAUI*FTR*(1.d0-ft)*GI*2.0_RP/LB*grkvot*GAV0       
            PI  = TVR*XIH*FT*GI*(A2+DIVGA2+DIVA2*WII)*2.0_RP/LB*grkvot 
            PIQ = - tvr*XIH*Z*H2*2.0_RP/LB*Bq*GI*grkvot 

c... effective diffusivity 
           RCI = (XI(1)+(XI(2)/LTe*Te+XI(3)*N/Ln+XI(4)*Tq/Ltq)*Lti/Ti 
     &          +XI(5)*NQ/Ti*LTI/LNQ)  - (PI+HP+PIQ)*LTI         
         ELSE 
C           --- JETTO implementation 
            XI(1) = 3.0_RP/2.0_RP*XIH 
            XI(2) = ft*XIH*TAUI*(B-DIVGA3-DIVGB- 
     &              (DIVA3+DIVB)*WII) 
            XI(3) = -XIH*(1.D0+ft*(A3+DIVGA1+DIVA1*WII))   !*(Ti/ni) 
            IF(NQ .GT. 1.0e-3_RP) THEN 
               XI(4) = -XIH*Z*K*TI/TQ 
               XI(5) = XIH*Z*(H1-ft*(A3+DIVGA1+DIVA1*WII)) !*(Ti/NI) 
            ENDIF 
 
c... Convective part of the ion heat flux 
            HP  = XIH*TAUI*FTR*(1.0_RP-ft)/(1.0_RP-BQ*Z)*2.0d0/LB*GAV0       
            PI  = XIH*FT/(1.0_RP-BQ*Z)*(A2+DIVGA2+DIVA2*WII)*2.0_RP/LB  
            PIQ =-XIH*H2*Bq*Z/(1.0_RP-BQ*Z)*2.0_RP/LB           
 
c... effective diffusivity 
            RCI = (XI(1)+(XI(2)/LTe*Te/(1.0_RP-BQ*Z)+ 
     &      Ti*XI(3)/Lni+XI(4)*Tq/Ltq*Bq/(1.0_RP-Bq*Z))*Lti/Ti 
     &      + XI(5)*Bq/(1.0_RP-Bq*Z)*LTI/LNQ)-(PI+HP+PIQ)*LTI !PISTEM *GRKVOT 
         END IF        
 
c----------------------------------------------------------------------- 
C Electron row of the transport matrix (heat) 
c----------------------------------------------------------------------- 
c        --- Chalmers implementation 
         IF (kform .eq. 0) THEN 
            XE(2) = ft*XEH*(1.0_RP+TVR*(DH-DEVGB-DEVGA3- 
     &              (DEVB+DEVA3)*WII)) 
            XE(3) = -TVR*FT*XEH*(C1+ DEVGA1+DEVA1*WII)*TE/N 
 
c... Convective part of the electron heat flux 
            PE = tvr*ft*XEH*(C2+DEVGA2+(DEVA2+PHS*VEF)*WII)*2.0_RP/LB 
     &            *grkvot 
 
c... effective diffusivity 
            RCE = XE(2) + XE(3)*N/LN*Lte/Te - PE*LTE 
 
         ELSE 
c        --- JETTO implementation 
            XE(2) = ft*XEH*(3.0_RP/2.0_RP+(DH-DEVGB-DEVGA3- 
     &             (DEVB+DEVA3)*WII)) 
            XE(3) =-ft*XEH*(C1+ DEVGA1+DEVA1*WII)     !*(TE/NE) 
            IF (NQ .gt. 1.0e-5_RP) THEN 
                XE(5) = -FT*XEH*(C1+ DEVGA1+DEVA1*WII)*Z 
            END IF 
  
c... Convective part of the electron heat flux 
            PE = ft*XEH*(C2+DEVGA2+(DEVA2+PHS*VEF)*WII)*2.0_RP/LB 
 
c... effective diffusivity 
            RCE =XE(2)+XE(3)/LNi*Lte*(1.0_RP-BQ*Z)+XE(5)*BQ/LNQ*LTE-
     &           PE*LTE !PISTEMP* grkvot 
         END IF 
c----------------------------------------------------------------------- 
c Particle (electron) row of the transport matrix 
c----------------------------------------------------------------------- 
c        --- Chalmers implementation 
         IF (kform .eq. 0) THEN  
            XD(2) = -ft*XDH*N/Te*(F+(SVB+SVAC3)*WII) 
            XD(3) = ft*XDH*(E1-SVAC1*WII) 
 
c... convective part of the particle flux 
            PN = -ft*XDH*2.0_RP/LB*(E2-SVAC2*WII)      *grkvot 
 
c... effective diffusivity 
            RD  = XD(2)*TE/LTE*LN/N + XD(3) - PN*LN 
 
         ELSE 
c        --- JETTO implementation 
            XD(2) = -ft*XDH*(F+(SVB+SVAC3)*WII)      !*N/Te 
            XD(3) = ft*XDH*(E1-SVAC1*WII) 
            XD(5) = ft*XDH*(E1-SVAC1*WII)*Z 
c... convective part of the particle flux 
             PN = -ft*XDH*2.0_RP/LB*(E2-SVAC2*WII)/(1.D0-BQ*Z) 
 
c... effective diffusivity 
            RD  = XD(2)/LTE*LNi+XD(3)+XD(5)*BQ/(1.0_RP-BQ*Z)/lnq*lni   
     &          - PN*LNi !PISTEMP *GRKVOT 
 
         END IF      
c----------------------------------------------------------------------- 
c Impurity Thermal Diffusivity in transport matrix 
c----------------------------------------------------------------------- 
c        --- Chalmers implementation 
         IF (kform .eq.0) THEN 
            IF (NQ.GT.1.d-7) THEN 
               XQ(4)  = (1.d0 + TVR*TS)*XQH 
               XQ(5)  = -tvr*TQ*(1.0_RP+T1)*XQH/NQ 
 
c... convective part of the impurity temperature flux 
               PQ = TVR*XQH*T2*2.0_RP/LB    * grkvot 
 
c... effective diffusivity 
               RCQ = XQ(4) +XQ(5)*NQ/TQ*LTQ/LNQ-PQ*LTQ*GRKVOT 
             END IF 
         ELSE 
c        --- JETTO implementation 
            IF (NQ.GT.1.d-3) THEN 
               XQ(4)  = 3.0_RP/2.0_RP*(1.d0 + TVR*TS)*XQH 
               XQ(5)  = -(1.0_RP+T1)*XQH                !*TQ/NQ 
 
c... convective part of the impurity temperature flux 
               PQ = XQH*T2*2.0_RP/LB  
 
c... effective diffusivity 
               RCQ = XQ(4) +XQ(5)*LTQ/LNQ-PQ*LTQ !PISTEMP*GRKVOT 
             END IF 
         END IF 
  
c----------------------------------------------------------------------- 
c Impurity Particle Diffusivity in transport matrix 
c----------------------------------------------------------------------- 
c        --- Chalmers implementation 
         IF (kform .eq.0) THEN 
            IF (NQ.GT.1.e-5_RP) THEN 
               XDQ(4) = -XDH*DQT*NQ/TQ 
               XDQ(5) = XDH*DQN1 
c   Add par. compression term DQN3, HN 23/9 2009
      

c... convective part of the impurity particle flux 
               PNQ = -XDH*(DQN2+DQN3)*2.0_RP/LB   * grkvot 
 
c... effective diffusivity 
               RNQ = XDQ(4)*TQ/LTQ*LNQ/NQ +XDQ(5)-PNQ*LNQ 
            ENDIF 
         ELSE 
c        --- JETTO implementation 
            IF (NQ.GT.1.e-5_RP) THEN 
               XDQ(4) = -XDH*DQT                   !*NQ/TQ 
               XDQ(5) = XDH*DQN1 
 
c... convective part of the impurity particle flux 
               PNQ = -XDH*(DQN2+DQN3)*2.0_RP/LB 
c... effective diffusivity 
               RNQ = XDQ(4)/LTQ*LNQ +XDQ(5)-PNQ*LNQ 
            ENDIF 
         END IF  
  
c----------------------------------------------------------------------- 
C Add the contributions from each excited mode 
c----------------------------------------------------------------------- 
c... Effective Difusivities 
         SCHI = SCHI + RCI 
         SCHE = SCHE + RCE 
         SD   = SD   + RD 
         SCHQ = SCHQ + RCQ 
         SDQ  = SDQ  + RNQ 

c... Diagonal contribution to main ion diffusivity
         XND  = XND  + XDH*(-NJR+(2.0_RP*WR+FTRT)*(wr+FTRT))/NJ2
 
c... Convective parts of the transport formulas 
         FL(1) = (FL(1) + PI + PIQ + HP) 
         FL(2) = FL(2) + PE 
         FL(3) = FL(3) + PN 
         FL(4) = (FL(4) + PQ) 
         FL(5) = (FL(5) + PNQ) 
 
c... Transport matrix 
         DO J = 1,5 
            CHI(J) = CHI(J) + XI(J) 
            CHE(J) = CHE(J) + XE(J)  
            D  (J) = D  (J) + XD(J) 
            CHQ(J) = CHQ(J) + XQ(J) 
            DQ(J)  = DQ(J)  + XDQ(J) 
         END DO 
 2000    CONTINUE 
      END DO 
  


c----------------------------------------------------------------------- 
c Clean up and return 
c----------------------------------------------------------------------- 
  
      DO I = 1,11 
        ZRP(I) = 0.0_RP
      END DO 
 
c----------------------------------------------------------------------- 
c Contributions to transport coefficients from finite beta and  
C collisions on free electrons.
c----------------------------------------------------------------------- 
 
      
        IF(NEQ.GE.9) THEN 
c 
        DT=D1*SQRT(TE)**3
        ETE=2.0_RP*LTE/LB 
c 
        SHPE=0.0_RP 
        SCHEF=0.0_RP
        DEF=0.0_RP 
        DMI =0.0_RP
        DMIT=0.0_RP
        DMIP = 0.0_RP
        D_MD=0.0_RP
c
        DO 199 J=1,NEQ 
          WR=DREAL(ZZ(J)) 
          WI=DIMAG(ZZ(J))-ABS(GSHEAR)
 
          IF(WI.LT. 1.0e-3_RP) GO TO 199 
c
          W=DCMPLX(WR,WI)  
          WSQ=WR*WR + WI*WI 

          DNI=(WR+FTR*TAUI)**2+WI*WI 
          XDH=DT*WI**3/RFL 
          XIH=XDH/DNI 
c------------------------------------------------------------------ 
c   HERE THE EIGENVECTORS FROM THE LINEAR SOLVER ARE IDENTIFIED  
c 
          FIH=DCMPLX(ZVR(1,J),ZVI(1,J)) ! Complex electrost. pot. 
c
          IF(NEQ.EQ.10) THEN 
            NEF=DCMPLX(ZVR(4,J),ZVI(4,J)) ! Complex free electr. dens. 
            TEF=DCMPLX(ZVR(6,J),ZVI(6,J)) ! Complex free electr. temp.           
          ELSE
            IJ=8   !!! Corresponds to NEQ=9 
            IF(NEQ.EQ.11) IJ=9 
            AV=DCMPLX(ZVR(IJ,J),ZVI(IJ,J))  !! Vector pot  
            NEF=FIH-(ZZ(J)-ENI)*AV/kpc 
            TEF=EE*ENI*AV/kpc
          ENDIF
 
c THE EIGENVECTORS HAVE BEEN OBTAINED  ------------------------ 
c 
          IF(CDABS(FIH).LT.0.0001D0) FIH=(0.0001D0,0.D0) 
          NRAT=NEF/FIH 
          ELN=NRAT-1.D0 
          ELNR=DREAL(ELN) 
          ELNI=DIMAG(ELN) 
          AINF=TVR*(FTR*TAUI*ELNR+ELNI*(WI+WR*(WR+FTR*TAUI))/WI) 
          HPE=XIH*GI*(1.0_RP-FT)*AINF*2.0_RP/lb
c 
          FRAT=TEF/FIH 
          IMF=-DIMAG(FRAT) 
          CEFT=(1.0_RP-FT)*IMF*ETE*XDH/WI 
c 
          GEI=-IMAG(NRAT)/WI 
          DEFT=(1.0_RP-FT)*GEI*EN*XDH 
c 
          SHPE=SHPE+HPE 
          SCHEF=SCHEF+CEFT 
          DEF=DEF+DEFT 

c ----Momentum transport ------
c

          TII=DCMPLX(ZVR(2,J),ZVI(2,J))
          NIF=DCMPLX(ZVR(3,J),ZVI(3,J))
          AVRAT=AV/FIH
          MRT=(TII+NIF)/FIH
          DMS=WI*WI*(1.0_RP+TAUI*DREAL(MRT))
          DMI=DMI+DMS
          ELMS= (W+(1.0_RP+EI)*TAUI/(EN*W))*AVRAT   !! EM part of v_par 
          DMSP=WI*WI*(IU/(w+2.0_RP*TAUI*GAV0))*(1+TAUI*MRT-EM*ELMS)
c---------------------
c  Calculation of effective Kparallel
          IF ( ABS(LVFT) .gt. 1.e-7_RP) THEN
            Kkap = Q*LB*RFL*VTOR/LVFT !Normering med <grad | rho|> saknas?
          ELSE
            KKAP = 0.0_RP   ! lim v/lvf --> 0
          END IF
          IF ( ABS(LVFP) .gt. 1.e-7_RP) THEN
             KAP1 = LN*VPOL/(LVFP*S*RFL)
          ELSE
             KAP1 = 0.0_RP   ! lim v/lvfp --> 0
          END IF

          Fm=W*(1.0_RP+FTRT)+TAUI*(EI-TVR)/EN
          Hm=FTRT*(1.0_RP+TAUI)
          hf=4.0_RP*RFL*RFL*Q*Q*W/EN
          KS=-2.0_RP/TAUI/TAUI*VTOR*RFL*Q
          KPF=-(0.5_RP*(W+FTR)*(Kkap+KS)+IU*hf*WI*KAP1)/((Fm+Hm)*Q*LB)
          KPF=KPF*CS/OM_DE
          DMST=BP2BT*DMS+DREAL(KPF*DMSP)/RFL
          DMIT=DMIT+DMST

          GP1 = -WI*WI*WI*GAV0/RFL/((WR+2.D0*TAUI*GAV0)**2+WI*WI)
          GP2 = WI*WI*GAV0/RFL* DREAL(IU*(MRT-EM*ELMS/TAUI)/
     &          (W+2.0_RP*TAUI*GAV0))
c

          DMIP = DMIP + GP1+ GP2
          D_MD= D_MD + D1*(SQRT(TE)*WI)**3/
     &          ((wr+2.0_RP*TAUI*GAV0)**2+WI**2)/RFL 

20199 FORMAT(' KPS=', G11.3,' Kkap=',G11.3,' KAP1=',G11.3,
     & ' Fm=',2G11.3,' Hm=',G11.3,
     & ' hf=',2G11.3,' KPF=',2G11.3,' DMST=', G11.3, 'DMSP=', 2G11.3)

  199 CONTINUE 
c 
c--------------------------------------------------------------- 
      IF (KFORM .EQ.0) THEN  ! --- CHALMERS MODEL 
         FL(1)=FL(1)+SHPE*grkvot    ! Add SHPE to the pinch part of CHII 
         CHE(2)=CHE(2)+SCHEF        ! Add SCHEF to the diagonal element of CHE 
         D(3)=D(3)+DEF              ! Add DEF to the diagonal element of D
         SCHI = SCHI - SHPE*grkvot*LTI
         SCHE = SCHE + SCHEF
         SD   = SD   + DEF  
      ELSE                   !--- JETTO MODEL 
         FL(1)=FL(1)+3.0_RP/2.0_RP*SHPE

         IF (SCHEF .LT. 0.0_RP) THEN 
            FL(2)= FL(2) - 3.0_RP/2.0_RP*SCHEF/LTE
         ELSE
            CHE(2)=CHE(2)+3.0_RP*SCHEF/2.0_RP   ! PS changed May03
         END IF
         If (DEF .LT. 0.0_RP) THEN
            FL(3) = FL(3)  - DEF/LN
         ELSE 
            D(3)=D(3)+DEF      
         END IF           

         SCHI = SCHI - SHPE*LTI
         SCHE = SCHE + SCHEF
         SD   = SD   + DEF


      END IF 

      IF (abs(VTOR) .gt. 1.e-7_RP) THEN
        DMTEF=2.0_RP*DT*(DMIT/VTOR+DMIP)*LVFT + D_MD !PS2008-04-09
        DMA=2.0_RP*DT*DMIT/VTOR*LVFT !PS2008-11-19
        DMB=2.0_RP*DT*DMIP*LVFT      !PS2008-11-19
        DMC=D_MD                   !PS2008-11-19       
      ELSE
        DMTEF = 0.0_RP
      END IF

 
      END IF 
      SDI = LNI/G*(SD/LN-Z*BQ*SDQ/LNQ)

      IF (KFORM .le. 1) THEN  !PisTEMP
!      IF (KFORM .eq. 0) THEN 
        XDPINCH(1) = (schi-Chi(1))/LTI*grkvot
        XDPINCH(2) = (sche-CHE(2))/LTE*grkvot
        XDPINCH(3) = (sd-D(3))/LN*grkvot
        XDPINCH(4) = (schq-CHQ(4))/LTQ*grkvot
        XDPINCH(5) = (sdq-DQ(5))/LNQ*grkvot
        XDPINCH(6) = (sdi-XND)/LNI*grkvot
      ELSE     
        XDPINCH(1) = (schi-Chi(1))/LTI
        XDPINCH(2) = (sche-CHE(2))/LTE
        XDPINCH(3) = (sd-D(3))/LN
        XDPINCH(4) = (schq-CHQ(4))/LTQ
        XDPINCH(5) = (sdq-DQ(5))/LNQ
        XDPINCH(6) = (sdi-XND)/LNI
      END IF
      XDDIA(1) = CHI(1)
      XDDIA(2) = CHE(2)
      XDDIA(3) = D(3)
      XDDIA(4) = CHQ(4)
      XDDIA(5) = DQ(5)
      XDDIA(6) = XND    

      RETURN 
      END SUBROUTINE
  
  
  
