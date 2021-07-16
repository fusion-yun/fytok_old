#ifdef MATLAB73
#include "fintrf.h"
#endif

      subroutine mexFunction(nlhs,plhs,nrhs,prhs)
C--------------------------------------------------------------
C Interface mex-file (matlab 5) pour GLF 23 (version 1.61 du 12 mars 2003)
C F. Imbeaux - Juillet 2003
c 
c Liste des modifications
c 12/09/2003 : angrot_exp est fournie en entree par Cronos (prof.vtor_exp)
c 24/02/2004 : passage a 26 points de grille (jpd = 25)
c 08/04/2004 : ajout de la sortie de la diffusivite de l'impurete (diffz)
c 04/05/2004 : option impur pour choisir nroot et idengrad
c 13/05/2004 : renormalisation des taux de croissance et shear rates avec csda_m
c 10/09/2004 : connexion des ions rapides (ns_m lue en entree)
c 20/05/2005 : remplissage par defaut de bteff_exp �1.0 DEPLAC�: defini comme un tableau --> toutes les cases doivent etre remplies (ecriture precedente simplement bteff_exp = 1.0 pas coherente)
c 
c Compile this mexfile on Deneb using : /usr/local/matlab5/bin/mex -f mexopts_glf.sh zglfmex_essai100.f blas_zgeev.o callglf2d.o  glf2d.o r8tomsqz.o zgeev.o xtverb.o
c Compile this mexfile on Jac using (inside matlab) : mex -f ../../../../../jetmexopts.sh zglfmex.f blas_zgeev.o callglf2d.o  glf2d.o r8tomsqz.o zgeev.o xtverb.o
c Compile this mexfile on PC-Cronos using (inside matlab) : mex -f ../../../../../intelmexopts.sh zglfmex.f blas_zgeev.o callglf2d.o  glf2d.o r8tomsqz.o zgeev.o xtverb.o
c
c
c for compiling the GLF23 subroutines on Deneb :
c setenv CPU DEC
c gmake
c
c for compiling the GLF23 subroutines on Jac : 
c rm *.o
c rm xtverb
c rm zglfmex.mexaxp
c export CPU = PGI
c gmake
c
c for compiling the GLF23 subroutines on PC-Cronos : 
c /home/cronos/Cronos/translinux.sh zglfmex.f 
c /home/cronos/Cronos/translinux.sh blas_zgeev.f
c /home/cronos/Cronos/translinux.sh callglf2d.f
c /home/cronos/Cronos/translinux.sh glf2d.F
c /home/cronos/Cronos/translinux.sh r8tomsqz.F
c /home/cronos/Cronos/translinux.sh zgeev.f
c /home/cronos/Cronos/translinux.sh xtverb.f
c mettre en commentaire la ligne 65 de xtverb.f : call exit(0)
c gmake clean
c csh
c setenv CPU INTEL
c gmake
c bash
c
c Reminder : use `gmake clean' to remove all .o files


      implicit none
c
c...declare variables
c
      character cdate*24, line*132
      integer jpd
 
      double precision epsilon
      parameter ( jpd=100 )
c jpd : number of radial points 
c Remarque : jpd = 50 dans la version originale de Kinsey      
c

C DECLARATION DES VARIABLES
c pointeurs mex   
      CRONOSINT plhs(*), prhs(*)
      
      MATLABINOUTDIM nlhs,nrhs
c variables de sortie      
c      real*8 VP(NRMMAX),C2(NRMMAX),C3(NRMMAX),R2M(NRMMAX),RHO(NRMMAX)
c pointeurs et tailles      
      CRONOSINT bt_exp_pr, rmajor_exp_pr, amassgas_exp_pr
      CRONOSINT zimp_exp_pr,arho_exp_pr, te_m_pr, ti_m_pr, ne_m_pr
      CRONOSINT zeff_exp_pr, rho_pr, gradrho_exp_pr, gradrhosq_exp_pr
      CRONOSINT rmin_exp_pr, rmaj_exp_pr, q_exp_pr, shat_exp_pr
      CRONOSINT elong_exp_pr, vphi_m_pr, vpar_m_pr, vper_m_pr
      CRONOSINT diff_m_pr, chie_m_pr, chii_m_pr, etaphi_m_pr
      CRONOSINT etaper_m_pr, exch_m_pr, egamma_m_pr, egamma_d_pr
      CRONOSINT gamma_p_m_pr, anrate_m_pr, anrate2_m_pr, anfreq_m_pr
      CRONOSINT anfreq2_m_pr, ni_m_pr, alpha_exp_pr, amassimp_exp_pr
      CRONOSINT etapar_m_pr, alpha_e_pr, x_alpha_pr, angrotp_exp_pr
      CRONOSINT diffz_m_pr, impur_pr, ns_m_pr
      integer   mxGetM, mxGetN
      CRONOSINT mxCreateDoubleMatrix, mxGetPr


      double precision te_m(0:jpd), ti_m(0:jpd)
     & , ne_m(0:jpd), ni_m(0:jpd), ns_m(0:jpd)
     & , zpte_m(0:jpd), zpti_m(0:jpd), zpne_m(0:jpd), zpni_m(0:jpd)
     & , angrotp_exp(0:jpd), egamma_exp(0:jpd), gamma_p_exp(0:jpd)
     & , vphi_m(0:jpd), vpar_m(0:jpd), vper_m(0:jpd)
     & , zeff_exp(0:jpd), bt_exp, bteff_exp(0:jpd), rho(0:jpd), arho_exp
     & , gradrho_exp(0:jpd), gradrhosq_exp(0:jpd)
     & , rmin_exp(0:jpd), rmaj_exp(0:jpd), rmajor_exp
     & , q_exp(0:jpd), shat_exp(0:jpd), alpha_exp(0:jpd)
     & , elong_exp(0:jpd), zimp_exp, amassimp_exp, amassgas_exp
     & , alpha_e, x_alpha, impur
c
      double precision zpte_in, zpti_in, zpne_in, zpni_in, drho
c
      double precision diffnem, chietem, chiitim
     & , etaphim, etaparm, etaperm, exchm
     & , diff_m(0:jpd), chie_m(0:jpd), chii_m(0:jpd), etaphi_m(0:jpd)
     & , etapar_m(0:jpd), etaper_m(0:jpd), exch_m(0:jpd)
     & , egamma_m(0:jpd), egamma_d(0:jpd,10), gamma_p_m(0:jpd)
     & , anrate_m(0:jpd), anrate2_m(0:jpd)
     & , anfreq_m(0:jpd), anfreq2_m(0:jpd), diffz_m(0:jpd)
     & , csda_m(0:jpd)
c
      integer lprint, nroot, jshoot, jmm, jmaxm, itport_pt(1:5)
     & , igrad, idengrad, i_delay, j, k, leigen, irotstab, bt_flag, iglf
c
      namelist /nlglf/ leigen, lprint, nroot, iglf, jshoot, jmm, jmaxm
     & , itport_pt, irotstab, te_m, ti_m, ne_m, ni_m, ns_m
     & , igrad, idengrad, zpte_in, zpti_in, zpne_in, zpni_in
     & , angrotp_exp, egamma_exp, gamma_p_exp, vphi_m, vpar_m, vper_m
     & , zeff_exp, bt_exp, bt_flag, rho, arho_exp
     & , gradrho_exp, gradrhosq_exp
     & , rmin_exp, rmaj_exp, rmajor_exp, q_exp, shat_exp
     & , alpha_exp, elong_exp, zimp_exp, amassimp_exp, amassgas_exp
     & , alpha_e, x_alpha
C---------------------------------------------------------------------
C GESTION DES ENTREES     
C toroidal field (T)
      bt_exp_pr = mxGetPr(prhs(1))
      call mxCopyPtrToReal8(bt_exp_pr,bt_exp,1)
C geometrical radius of magnetic axis (m)
      rmajor_exp_pr = mxGetPr(prhs(2))
      call mxCopyPtrToReal8(rmajor_exp_pr,rmajor_exp,1)
C atomic mass of working gas
      amassgas_exp_pr = mxGetPr(prhs(3))
      call mxCopyPtrToReal8(amassgas_exp_pr,amassgas_exp,1)
C average atomic mass of impurity
      amassimp_exp_pr = mxGetPr(prhs(4))
      call mxCopyPtrToReal8(amassimp_exp_pr,amassimp_exp,1)
C average charge of impurity
      zimp_exp_pr = mxGetPr(prhs(5))
      call mxCopyPtrToReal8(zimp_exp_pr,zimp_exp,1)
C toroidal flux coordinate at last closed flux surface (m)
      arho_exp_pr = mxGetPr(prhs(6))
      call mxCopyPtrToReal8(arho_exp_pr,arho_exp,1)
C Te profile (keV)
      te_m_pr = mxGetPr(prhs(7))
      call mxCopyPtrToReal8(te_m_pr,te_m,jpd+1)
C Ti profile (keV)
      ti_m_pr = mxGetPr(prhs(8))
      call mxCopyPtrToReal8(ti_m_pr,ti_m,jpd+1)
C ne profile (10^19 m-3)
      ne_m_pr = mxGetPr(prhs(9))
      call mxCopyPtrToReal8(ne_m_pr,ne_m,jpd+1)
C ni profile (10^19 m-3)
      ni_m_pr = mxGetPr(prhs(10))
      call mxCopyPtrToReal8(ni_m_pr,ni_m,jpd+1)
C Zeff profile
      zeff_exp_pr = mxGetPr(prhs(11))
      call mxCopyPtrToReal8(zeff_exp_pr,zeff_exp,jpd+1)
C normalised toroidal flux coordinate profile
      rho_pr = mxGetPr(prhs(12))
      call mxCopyPtrToReal8(rho_pr,rho,jpd+1)
C <|grad(rho)|>
      gradrho_exp_pr = mxGetPr(prhs(13))
      call mxCopyPtrToReal8(gradrho_exp_pr,gradrho_exp,jpd+1)
C <|grad(rho)|^2>
      gradrhosq_exp_pr = mxGetPr(prhs(14))
      call mxCopyPtrToReal8(gradrhosq_exp_pr,gradrhosq_exp,jpd+1)
C local minor radius (m)
      rmin_exp_pr = mxGetPr(prhs(15))
      call mxCopyPtrToReal8(rmin_exp_pr,rmin_exp,jpd+1)
C local major radius (m)
      rmaj_exp_pr = mxGetPr(prhs(16))
      call mxCopyPtrToReal8(rmaj_exp_pr,rmaj_exp,jpd+1)
C safety factor
      q_exp_pr = mxGetPr(prhs(17))
      call mxCopyPtrToReal8(q_exp_pr,q_exp,jpd+1)
C magnetic shear d ln q/ d ln rho
      shat_exp_pr = mxGetPr(prhs(18))
      call mxCopyPtrToReal8(shat_exp_pr,shat_exp,jpd+1)
C MHD alpha -q^2 R (dbeta/dr)
      alpha_exp_pr = mxGetPr(prhs(19))
      call mxCopyPtrToReal8(alpha_exp_pr,alpha_exp,jpd+1)
C local elongation
      elong_exp_pr = mxGetPr(prhs(20))
      call mxCopyPtrToReal8(elong_exp_pr,elong_exp,jpd+1)
C toroidal velocity (m/s)
      vphi_m_pr = mxGetPr(prhs(21))
      call mxCopyPtrToReal8(vphi_m_pr,vphi_m,jpd+1)
C paralell velocity (m/s)
      vpar_m_pr = mxGetPr(prhs(22))
      call mxCopyPtrToReal8(vpar_m_pr,vpar_m,jpd+1)
C perpendicular velocity (m/s)
      vper_m_pr = mxGetPr(prhs(23))
      call mxCopyPtrToReal8(vper_m_pr,vper_m,jpd+1)
C experimental toroidal angular velocity (rad/s)
      angrotp_exp_pr = mxGetPr(prhs(24))
      call mxCopyPtrToReal8(angrotp_exp_pr,angrotp_exp,jpd+1)
C alpha_e : coefficient / switch of the ExB stabilisation (1.35 in retuned model)
      alpha_e_pr = mxGetPr(prhs(25))
      call mxCopyPtrToReal8(alpha_e_pr,alpha_e,1)
C x_alpha : switch for alpha stabilisation (0 = off, 1. = on, -1. for self consistent alpha stabilisation)
      x_alpha_pr = mxGetPr(prhs(26))
      call mxCopyPtrToReal8(x_alpha_pr,x_alpha,1)
C impur : switch for impurities (tunes idengrad, nroot, ...)
      impur_pr = mxGetPr(prhs(27))
      call mxCopyPtrToReal8(impur_pr,impur,1)
C ns profile (fast ion density) (10^19 m-3)
      ns_m_pr = mxGetPr(prhs(28))
      call mxCopyPtrToReal8(ns_m_pr,ns_m,jpd+1)

  
c
c..default inputs (not filled by Cronos)
c
      epsilon  = 1.e-10
      leigen   = 0   ! for cgg eigenvalue solver
      if (impur.eq.0.D0) then
          nroot = 8   ! number of roots in eigenvalue solver,8 for default, 12 for impurity dynamics
          idengrad = 2  ! simple dilution
      else
          nroot = 12   ! number of roots in eigenvalue solver,8 for default, 12 for impurity dynamics
          idengrad = 3  ! one main hydrogenic ion + one impurity;PB idengrad = 3 GIVES PROBLEMS TO THE GLF23 SOLVER ???
      endif
      iglf     = 1   ! retuned GLF23 normalization
      jshoot   = 0   ! for time-dependent code
      jmaxm    = jpd ! point de grille max ou on calcule les coefficients de transport
      jmm      = 0   ! use arrays for input/output (see GLF23 notice)
      igrad    = 0   ! compute gradients
      i_delay  = 0
      itport_pt(1) = 1 ! density transport
      itport_pt(2) = 1 ! electron heat transport 
      itport_pt(3) = 1 ! ion heat transport
      itport_pt(4) = 0 ! vphi transport
      itport_pt(5) = 0 ! vtheta transport
      irotstab     = 1    ! use internally computed ExB shear, 0 for prescribed
c      bt_exp       = 1.0
      bt_flag      = 1    ! use effective B-field (real geometry)
c      rmajor_exp   = 1.0
c      amassgas_exp = 1.0
c      zimp_exp     = 6.0
c      amassimp_exp = 12.0
c      arho_exp     = 1.0
c      alpha_e      = 1.35   ! ExB shear stabilization, according to page 2 (1.35 in retuned model)
c      x_alpha      = 1.   ! alpha stabilization
      zpte_in      = 0.
      zpti_in      = 0.
      zpne_in      = 0.
      zpni_in      = 0.
c
      do j=0,jpd
c        te_m(j)   = 0.0
c        ti_m(j)   = 0.0
c        ne_m(j)   = 0.0
c        ni_m(j)   = 0.0
c        ns_m(j)   = 0.0  ! densite ions rapides
c
c        write(*,*) ni_m(j),ns_m(j)
c gradients calcules et non prescrits (igrad = 0) 
        zpte_m(j) = 0.0
        zpti_m(j) = 0.0
        zpne_m(j) = 0.0
        zpni_m(j) = 0.0
c
        bteff_exp(j) = 1.0  ! effective B-field (used when bt_flag > 0), dummy ?
c        angrotp_exp(j)   = 0.0
        egamma_exp(j)    = 0.0
        gamma_p_exp(j)   = 0.0
c        vphi_m(j)        = 0.0
c        vpar_m(j)        = 0.0
c        vper_m(j)        = 0.0
c
c        zeff_exp(j)   = 1.0
c        rho(j)        = 0.0
c        gradrho_exp(j)   = 1.0
c        gradrhosq_exp(j) = 1.0
c        rmin_exp(j)   = 0.0
c        rmaj_exp(j)   = 0.0
c        q_exp(j)      = 1.0
c        shat_exp(j)   = 0.0
c        alpha_exp(j)  = 0.0
c        elong_exp(j)  = 1.0
      enddo
c
c..default outputs
c
      diffnem      = 0
      chietem      = 0
      chiitim      = 0
      etaphim      = 0
      etaparm      = 0
      etaperm      = 0
      exchm        = 0
c
      do j=0,jpd
        diff_m(j)   = 0.0
        diffz_m(j)  = 0.0
        chie_m(j)   = 0.0
        chii_m(j)   = 0.0
        etaphi_m(j) = 0.0
        etapar_m(j) = 0.0
        etaper_m(j) = 0.0
        exch_m(j)   = 0.0
        egamma_m(j) = 0.0
        gamma_p_m(j)= 0.0
        anrate_m(j) = 0.0
        anrate2_m(j) = 0.0
        anfreq_m(j) = 0.0
        anfreq2_m(j) = 0.0
        do k=1,10
          egamma_d(j,k) = 0.0
        enddo
      enddo
c
c------------ MAIN CALL ----------------------------------------------
c
        call callglf2d( leigen, nroot, iglf
     & , jshoot, jmm, jmaxm, itport_pt
     & , irotstab, te_m, ti_m, ne_m, ni_m, ns_m
     & , igrad, idengrad, zpte_in, zpti_in, zpne_in, zpni_in
     & , angrotp_exp, egamma_exp, gamma_p_exp, vphi_m, vpar_m, vper_m
     & , zeff_exp, bt_exp, bt_flag, rho
     & , arho_exp, gradrho_exp, gradrhosq_exp
     & , rmin_exp, rmaj_exp, rmajor_exp, zimp_exp, amassimp_exp
     & , q_exp, shat_exp, alpha_exp, elong_exp, amassgas_exp
     & , alpha_e, x_alpha, i_delay
     & , diffnem, chietem, chiitim, etaphim, etaparm, etaperm
     & , exchm, diff_m, diffz_m, chie_m, chii_m, etaphi_m, etapar_m
     & , etaper_m, exch_m, egamma_m, egamma_d, gamma_p_m
     & , anrate_m, anrate2_m, anfreq_m, anfreq2_m, csda_m )
c

      do j=1,jmaxm
c        drho=rho(j-1)-rho(j)+epsilon
c        zpte_m(j)=-(log(te_m(j-1))-log(te_m(j)))/drho
c        zpti_m(j)=-(log(ti_m(j-1))-log(ti_m(j)))/drho
c        zpne_m(j)=-(log(ne_m(j-1))-log(ne_m(j)))/drho
c        zpni_m(j)=-(log(ni_m(j-1))-log(ni_m(j)))/drho

c Renormalisation into 1/s (F.I. 13 May 2004)
         anrate_m(j) = anrate_m(j) * csda_m(j)
         anfreq_m(j) = anfreq_m(j) * csda_m(j)
         anrate2_m(j) = anrate2_m(j) * csda_m(j)
         anfreq2_m(j) = anfreq2_m(j) * csda_m(j)
         egamma_m(j) = egamma_m(j) * csda_m(j)
c         do k=1,10
c            egamma_d(j,k) = egamma_d(j,k) * csda_m(j)
c         enddo
         gamma_p_m(j) = gamma_p_m(j) * csda_m(j)
      enddo
c
c
c ----------- OUTPUTS ---------
C diff_m : particle (ion) diffusivity (m^2/s)
      plhs(1) = mxCreateDoubleMatrix(jpd+1,1,0)
      diff_m_pr = mxGetPr(plhs(1))
      call mxCopyReal8ToPtr(diff_m,diff_m_pr,jpd+1)
C chie_m : electron thermal diffusivity (m^2/s)
      plhs(2) = mxCreateDoubleMatrix(jpd+1,1,0)
      chie_m_pr = mxGetPr(plhs(2))
      call mxCopyReal8ToPtr(chie_m,chie_m_pr,jpd+1)
C chii_m : ion thermal diffusivity (m^2/s)
      plhs(3) = mxCreateDoubleMatrix(jpd+1,1,0)
      chii_m_pr = mxGetPr(plhs(3))
      call mxCopyReal8ToPtr(chii_m,chii_m_pr,jpd+1)
C etaphi_m : toroidal velocity diffusivity (m^2/s)
      plhs(4) = mxCreateDoubleMatrix(jpd+1,1,0)
      etaphi_m_pr = mxGetPr(plhs(4))
      call mxCopyReal8ToPtr(etaphi_m,etaphi_m_pr,jpd+1)
C etapar_m : paralell velocity diffusivity (m^2/s)
      plhs(5) = mxCreateDoubleMatrix(jpd+1,1,0)
      etapar_m_pr = mxGetPr(plhs(5))
      call mxCopyReal8ToPtr(etapar_m,etapar_m_pr,jpd+1)
C etaper_m : perpendicular ion diffusivity (m^2/s)
      plhs(6) = mxCreateDoubleMatrix(jpd+1,1,0)
      etaper_m_pr = mxGetPr(plhs(6))
      call mxCopyReal8ToPtr(etaper_m,etaper_m_pr,jpd+1)
C exch_m : turbulent electron-ion energy exchange (MW/m^3)
      plhs(7) = mxCreateDoubleMatrix(jpd+1,1,0)
      exch_m_pr = mxGetPr(plhs(7))
      call mxCopyReal8ToPtr(exch_m,exch_m_pr,jpd+1)
C egamma_m : ExB shear rate in units of csda_m (1/s)
      plhs(8) = mxCreateDoubleMatrix(jpd+1,1,0)
      egamma_m_pr = mxGetPr(plhs(8))
      call mxCopyReal8ToPtr(egamma_m,egamma_m_pr,jpd+1)
C egamma_d : ExB shear rate delayed by i-delays steps
      plhs(9) = mxCreateDoubleMatrix(jpd+1,1,0)
      egamma_d_pr = mxGetPr(plhs(9))
      call mxCopyReal8ToPtr(egamma_d,egamma_d_pr,jpd+1)
C gamma_p_m : parallel velocity shear rate in units of local csda_m (1/s)
      plhs(10) = mxCreateDoubleMatrix(jpd+1,1,0)
      gamma_p_m_pr = mxGetPr(plhs(10))
      call mxCopyReal8ToPtr(gamma_p_m,gamma_p_m_pr,jpd+1)
C anrate_m : growth rate of leading mode in units of local csda_m (1/s)
      plhs(11) = mxCreateDoubleMatrix(jpd+1,1,0)
      anrate_m_pr = mxGetPr(plhs(11))
      call mxCopyReal8ToPtr(anrate_m,anrate_m_pr,jpd+1)
C anrate2_m : growth rate of second leading mode in units of local csda_m (1/s)
      plhs(12) = mxCreateDoubleMatrix(jpd+1,1,0)
      anrate2_m_pr = mxGetPr(plhs(12))
      call mxCopyReal8ToPtr(anrate2_m,anrate2_m_pr,jpd+1)
C anfreq_m : frequency of leading mode in units of local csda_m (1/s)
      plhs(13) = mxCreateDoubleMatrix(jpd+1,1,0)
      anfreq_m_pr = mxGetPr(plhs(13))
      call mxCopyReal8ToPtr(anfreq_m,anfreq_m_pr,jpd+1)
C anfreq2_m : frequency of second leading mode in units of local csda_m (1/s)
      plhs(14) = mxCreateDoubleMatrix(jpd+1,1,0)
      anfreq2_m_pr = mxGetPr(plhs(14))
      call mxCopyReal8ToPtr(anfreq2_m,anfreq2_m_pr,jpd+1)
C diffz_m : impurity particle diffusivity (m^2/s)
      plhs(15) = mxCreateDoubleMatrix(jpd+1,1,0)
      diffz_m_pr = mxGetPr(plhs(15))
      call mxCopyReal8ToPtr(diffz_m,diffz_m_pr,jpd+1)


      return
      end
