<?xml version="1.0" encoding="UTF-8"?>
<!--  
  Generate Python class (FyTok IDS) from IDSDef.xml file 
  
  copyright:
     @ASIPP, 2023,

  authors:
     Zhi YU, @ASIPP

  changes:
    2023-04-26: 0.0.1, ZY, initial from fy_imas.xsl
     

-->
<xsl:stylesheet  
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions"	
  xmlns:my="http://www.example.com/my"  
  xmlns:saxon="http://saxon.sf.net/"
  version="3.0"
>
<!-- Declare element ######################################################################################### -->

<xsl:variable name="type_map">
    <entry key='STR_0D'       >str</entry>
    <entry key='str_type'     >str</entry> 
    <entry key='STR_1D'       >List[str]</entry>
    <entry key='str_1d_type'  >List[str]</entry>
    <entry key='INT_0D'       >int</entry>
    <entry key='int_type'     >int</entry>
    <entry key='INT_1D'       >np.ndarray</entry>
    <entry key='int_1d_type'  >np.ndarray</entry>
    <entry key='INT_2D'       >np.ndarray</entry>
    <entry key='INT_3D'       >np.ndarray</entry>
    <entry key='INT_4D'       >np.ndarray</entry>
    <entry key='INT_5D'       >np.ndarray</entry>
    <entry key='INT_6D'       >np.ndarray</entry>
    <entry key='FLT_0D'       >float</entry>
    <entry key='flt_type'     >float</entry>
    <entry key='FLT_1D'       >np.ndarray</entry>
    <entry key='flt_1d_type'  >np.ndarray</entry>
    <entry key='FLT_2D'       >np.ndarray</entry>
    <entry key='FLT_3D'       >np.ndarray</entry>
    <entry key='FLT_4D'       >np.ndarray</entry>
    <entry key='FLT_5D'       >np.ndarray</entry>
    <entry key='FLT_6D'       >np.ndarray</entry>
    <entry key='cpx_type'     >complex</entry>
    <entry key='cplx_1d_type' >np.ndarray</entry>
    <entry key='CPX_0D'       >np.ndarray</entry>
    <entry key='CPX_1D'       >np.ndarray</entry>
    <entry key='CPX_2D'       >np.ndarray</entry>
    <entry key='CPX_3D'       >np.ndarray</entry>
    <entry key='CPX_4D'       >np.ndarray</entry>
    <entry key='CPX_5D'       >np.ndarray</entry>
    <entry key='CPX_6D'       >np.ndarray</entry>
    
    <entry key='signal_flt_1d'>Signal[float]</entry>
    <entry key='signal_flt_2d'>SignalND[float]</entry>
    <entry key='signal_flt_3d'>SignalND[float]</entry>
    <entry key='signal_flt_4d'>SignalND[float]</entry>
    <entry key='signal_flt_5d'>SignalND[float]</entry>
    <entry key='signal_flt_6d'>SignalND[float]</entry>

    <entry key='signal_int_1d'>Signal[int]</entry>
    <entry key='signal_int_2d'>SignalND[int]</entry>
    <entry key='signal_int_3d'>SignalND[int]</entry>
    <entry key='signal_int_4d'>SignalND[int]</entry>
    <entry key='signal_int_5d'>SignalND[int]</entry>
    <entry key='signal_int_6d'>SignalND[int]</entry>

    <entry key='code'>_T_Code</entry>
    <entry key='time'>np.ndarray</entry>



</xsl:variable>



</xsl:stylesheet>
