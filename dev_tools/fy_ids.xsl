<?xml version="1.0" encoding="UTF-8"?>
<!-- Zhi YU, ASIPP, 2023, Generating Python class (FyTok IDS) from IDSDef.xml file -->

<xsl:stylesheet  
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions"	
  xmlns:my="http://www.example.com/my"
  version="3.0"
>
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>

<xsl:param name="IDS_NAME" select="('distributions')" />

<xsl:param name="IDS_FIELDS" select="('ids_properties','code')" />

<xsl:param name="CURRENT_DATETIME" select="0-0-0" />

<xsl:param name="line-width" select="80" />
   

<xsl:function name="my:to-camel-case" as="xs:string">
  <xsl:param name="string" as="xs:string"/>
  <xsl:sequence select="string-join(tokenize($string, '_')!(upper-case(substring(., 1, 1)) || lower-case(substring(., 2))))"/>
</xsl:function>

<!-- <xsl:function name="my:line-wrap" as="xs:string">
  <xsl:param name="text" as="xs:string" />
  <xsl:param name="line-length" as="xs:integer" />
  <xsl:param name="indent" as="xs:integer" />
  <xsl:variable name="spaces" select="string-join((for $i in 1 to $indent return ' '), '')" />
  <xsl:sequence select="replace(concat(normalize-space($text),' '), concat('(.{0,', $line-length, '}) '), concat('$1&#10;', $spaces))" />
</xsl:function> -->

<xsl:function name="my:line-wrap" as="xs:string">
  <xsl:param name="text" as="xs:string" />
  <xsl:param name="line-length" as="xs:integer" />
  <xsl:param name="indent" as="xs:integer" />
  <xsl:variable name="spaces" select="string-join((for $i in 1 to $indent return ' '), '')" />
  <xsl:variable name="wrapped-text" select="replace(concat(normalize-space($text),' '), concat('(.{0,', $line-length, '}) '), concat('$1&#10;', $spaces))" />
  <xsl:sequence select="substring($wrapped-text, 1, string-length($wrapped-text) - $indent - 1)" />
</xsl:function>

<xsl:variable name="head_import" >"""
Generate by FyDev/fy_ids_generator
create_date: <xsl:value-of  select="$CURRENT_DATETIME" />

  IMAS data dictionary 
    version = <xsl:value-of select="/IDSs/version" /> 
    cocos   = <xsl:value-of select="/IDSs/cocos" /> 
"""
import numpy as np
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from spdm.util.logger import logger
from fytok.common.IDS import IDS  


</xsl:variable>


<!-- Directory:  _imas -->
<xsl:template match="/">

<xsl:apply-templates select="/IDSs/utilities" mode="FILE"/>   


<xsl:for-each select="/IDSs/IDS[contains($IDS_NAME,@name)]">
from .<xsl:value-of select="@name"/>  import _T_<xsl:value-of select="@name"/>

<xsl:apply-templates select="." mode="FILE"/>   

</xsl:for-each>

</xsl:template>


<!-- FILE:  utilities.py -->
<xsl:template match="IDSs/utilities" mode="FILE" >   
<xsl:result-document method="text" href="utilities.py">

<xsl:copy-of select="$head_import" />

<xsl:for-each select="field[(@data_type='structure' or @data_type='struct_array')]">

class _T_<xsl:value-of select="@name"/>(Dict[Node]):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""   
    <xsl:apply-templates select="field[substring(@name,string-length(@name)-string-length('_error_upper')+1)!='_error_upper'
    and substring(@name,string-length(@name)-string-length('_error_lower')+1)!='_error_lower'
    and substring(@name,string-length(@name)-string-length('_error_index')+1)!='_error_index'
    ]" mode = "DECLARE"/>
</xsl:for-each>

</xsl:result-document>
</xsl:template>

<!-- FILE:  {IDS/@name}.py -->

<xsl:template match = "IDSs/IDS" mode="FILE">   
<xsl:result-document method="text" href="{@name}.py">

<xsl:copy-of select="$head_import" />

<xsl:for-each select="field[@data_type='structure' or @data_type='struct_array']">
<xsl:variable name="structure_reference" select="@structure_reference"/>
<xsl:choose>
  <xsl:when test="/IDSs/utilities/field[@data_type='structure' and @name=$structure_reference]">
from .utilities import _T_<xsl:value-of select="$structure_reference"/>  
  </xsl:when>
  <xsl:otherwise>
    <xsl:apply-templates select="." mode = "DEFINE"/>   
  </xsl:otherwise>
</xsl:choose>
</xsl:for-each>

class _T_<xsl:value-of select="@name"/>(IDS):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""   

    _IDS = "<xsl:value-of select="@name"/>" 

    <xsl:for-each select="field" >
      <xsl:if test="@name!='ids_properties' and @name!='code'">
        <xsl:apply-templates select="." mode = "DECLARE"/>   
      </xsl:if>
    </xsl:for-each>
</xsl:result-document>     
</xsl:template>

<!-- Template for fields -->

<xsl:template match = "field" mode = "DECLARE"> 
<xsl:text>&#xA;    </xsl:text>
  <xsl:choose>
    <xsl:when test="@data_type='str_type'    or @data_type='STR_0D'">  <xsl:value-of select="@name"/>:str       = sp_property(type="<xsl:value-of select="@type"/>")    </xsl:when>
    <xsl:when test="@data_type='str_1d_type' or @data_type='STR_1D'">  <xsl:value-of select="@name"/>:List[str] = sp_property(type="<xsl:value-of select="@type"/>" )    </xsl:when>   

    <xsl:when test="@data_type='int_type'">       <xsl:value-of select="@name"/>:int        = sp_property(type="<xsl:value-of select="@type"/>") </xsl:when>    
    <xsl:when test="@data_type='int_1d_type'">    <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", ndims=1, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='INT_0D'">         <xsl:value-of select="@name"/>:int        = sp_property(type="<xsl:value-of select="@type"/>") </xsl:when>    
    <xsl:when test="@data_type='INT_1D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", ndims=1, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='INT_2D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", ndims=2, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='INT_3D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", ndims=3, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='flt_type'">       <xsl:value-of select="@name"/>:float      = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when> 
    <xsl:when test="@data_type='flt_1d_type'">    <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>       
    <xsl:when test="@data_type='FLT_0D'">         <xsl:value-of select="@name"/>:float      = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when> 
    <xsl:when test="@data_type='FLT_1D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_2D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=2, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_3D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=3, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_4D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=4, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_5D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=5, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />", coordinate5="<xsl:value-of select="@coordinate5" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_6D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=6, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />", coordinate5="<xsl:value-of select="@coordinate5" />", coordinate6="<xsl:value-of select="@coordinate6" />") </xsl:when>   
    <xsl:when test="@data_type='cpx_type'">       <xsl:value-of select="@name"/>:complex    = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when>   
    <xsl:when test="@data_type='cplx_1d_type'">   <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=complex, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>   
    <xsl:when test="@data_type='CPX_0D'">         <xsl:value-of select="@name"/>:complex    = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when>   
    <xsl:when test="@data_type='CPX_1D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=complex, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>   
    <xsl:when test="@data_type='CPX_2D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=2, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_3D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=3, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_4D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=4, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_5D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=5, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_6D'">         <xsl:value-of select="@name"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=6, data_type=complex) </xsl:when>   

    <xsl:when test="@data_type='structure'"><xsl:value-of select="@name"/>:_T_<xsl:value-of select="@structure_reference"/>  = sp_property() </xsl:when>
    <xsl:when test="@data_type='struct_array'"><xsl:value-of select="@name"/>:List[_T_<xsl:value-of select="@structure_reference"/>]   = sp_property(coordinate1="<xsl:value-of select="@coordinate1"/>") </xsl:when>
  </xsl:choose>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""
</xsl:template>

<xsl:template match = "field[@data_type='structure' or @data_type='struct_array']" mode = "DEFINE"> 

<xsl:for-each select="field[@data_type='structure' or @data_type='struct_array']">
<xsl:variable name="structure_reference" select="@structure_reference"/>
<xsl:choose>
  <xsl:when test="/IDSs/utilities/field[@data_type='structure' and @name=$structure_reference]">
from .utilities import _T_<xsl:value-of select="$structure_reference"/>  
  </xsl:when>
  <xsl:otherwise>
    <xsl:apply-templates select="." mode = "DEFINE"/>   
  </xsl:otherwise>
</xsl:choose>
</xsl:for-each>

class _T_<xsl:value-of select="@structure_reference"/>(Dict[Node]):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""

    <xsl:apply-templates select="field[substring(@name,string-length(@name)-string-length('_error_upper')+1)!='_error_upper'
    and substring(@name,string-length(@name)-string-length('_error_lower')+1)!='_error_lower'
    and substring(@name,string-length(@name)-string-length('_error_index')+1)!='_error_index'
    ]" mode = "DECLARE"/>
</xsl:template>
</xsl:stylesheet>
