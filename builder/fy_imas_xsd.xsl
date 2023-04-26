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
  version="3.0"
>
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>

<xsl:param name="FYTOK_REV" select="'0.0.1'" />

<xsl:param name="DD_GIT_DESCRIBE" as="xs:string" />

<xsl:param name="IMAS_DD_PATH" as="xs:string"  />

<!-- <xsl:param name="FILE_HEADER" select=" N/A " /> -->

<xsl:param name="line-width" select="80" />
   

<xsl:function name="my:to-camel-case" as="xs:string">
  <xsl:param name="string" as="xs:string"/>
  <xsl:sequence select="string-join(tokenize($string, '_')!(upper-case(substring(., 1, 1)) || lower-case(substring(., 2))))"/>
</xsl:function>

<xsl:function name="my:line-wrap" as="xs:string">
  <xsl:param name="text" as="xs:string" />
  <xsl:param name="line-length" as="xs:integer" />
  <xsl:param name="indent" as="xs:integer" />
  <xsl:variable name="spaces" select="string-join((for $i in 1 to $indent return ' '), '')" />
  <xsl:variable name="wrapped-text" select="replace(concat(normalize-space(translate($text, '&quot;', '_')),' '), concat('(.{0,', $line-length, '}) '), concat('$1&#10;', $spaces))" />
  <xsl:sequence select="substring($wrapped-text, 1, string-length($wrapped-text) - $indent - 1)" />
</xsl:function>

<xsl:function name="my:match_util" as="xs:string*">
  <xsl:param name="path" as="element()*"/>
  <xsl:for-each select="$path[(@data_type='structure' or @data_type='struct_array') and not(@doc_identifier)]">
    <xsl:variable name="structure_reference" select="@structure_reference"/>
    <xsl:choose>
    <xsl:when test="/IDSs/utilities/field[@name=$structure_reference]">
      <xsl:sequence select="$structure_reference"/>
    </xsl:when> 
    <xsl:otherwise>
      <xsl:sequence select="my:match_util(./field)"/>   
    </xsl:otherwise>     
    </xsl:choose>
  </xsl:for-each>
</xsl:function>

 

<xsl:function name="my:list_util" as="xs:string*">
  <xsl:param name="path" as="element()*"/>
  <xsl:for-each select="$path[(@data_type='structure' or @data_type='struct_array')]">
      <xsl:sequence select="@name"/>
      <xsl:sequence select="my:list_util(./field)"/>   
  </xsl:for-each>
</xsl:function>

<xsl:function name="my:dep_level" as="xs:integer">
  <xsl:param name="path" as="element()*"/>
  <xsl:variable name="children" select="for $sub_field in $path/field[(@data_type='structure' or @data_type='struct_array')] return my:dep_level($sub_field)"/>
  <xsl:choose>
  <xsl:when test="empty($children)"> <xsl:sequence select="0"/></xsl:when>
  <xsl:otherwise><xsl:sequence select="1+max($children)"/></xsl:otherwise>
  </xsl:choose>  
</xsl:function>

<xsl:function name="my:py_word">
  <xsl:param name="word"/>
  <xsl:variable name="keywords" select="'and,as,assert,break,class,continue,def,del,elif,else,except,False,finally,for,from,global,if,import,in,is,Lambda,None,nonlocal,not,or,pass,Raise,True,Try,while,yield'"/>
  <xsl:variable name="is-keyword" select="contains(concat(',', $keywords, ','), concat(',', $word, ','))"/>
  <xsl:variable name="word-with-underscores" select="translate($word, ' /', '__')"/>
  <xsl:value-of select="$word-with-underscores"/>
  <xsl:if test="$is-keyword">
    <xsl:text>_</xsl:text>
  </xsl:if>
</xsl:function>

<xsl:variable name="type_map">
    <entry key='STR_0D'       >str</entry>
    <entry key='STR_1D'       >List[str]</entry>
    <entry key='str_type'     >str</entry> 
    <entry key='str_1d_type'  >List[str]</entry>
    <entry key='INT_0D'       >int</entry>
    <entry key='INT_1D'       >List[int]</entry>
    <entry key='int_type'     >int</entry>
    <entry key='int_1d_type'  >List[int]</entry>
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
    <entry key='cpx_type'     >complex   </entry>
    <entry key='cplx_1d_type' >np.ndarray</entry>
    <entry key='CPX_0D'       >complex   </entry>
    <entry key='CPX_1D'       >np.ndarray</entry>
    <entry key='CPX_2D'       >np.ndarray</entry>
    <entry key='CPX_3D'       >np.ndarray</entry>
    <entry key='CPX_4D'       >np.ndarray</entry>
    <entry key='CPX_5D'       >np.ndarray</entry>
    <entry key='CPX_6D'       >np.ndarray</entry>
</xsl:variable>

<xsl:function name="my:type_hint">
  <xsl:param name="d" as="element()*"/>
  <xsl:variable name="t1">
    <xsl:choose>
      <xsl:when test="$d[@type]"><xsl:value-of select="$d/@type" /></xsl:when>
      <xsl:when test="$d[@ref]" ><xsl:value-of select="$d/@ref" /></xsl:when>
      <xsl:otherwise><xsl:value-of  select="$d/xs:complexType/xs:group/@ref" /> </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
  <xsl:variable name="t2">
    <xsl:choose>
      <xsl:when test="$type_map/entry[@key=$t1]"><xsl:value-of select="$type_map/entry[@key=$t1]" /> </xsl:when>
      <xsl:otherwise>_T_<xsl:value-of select="$t1"/></xsl:otherwise>
    </xsl:choose>
  </xsl:variable>  
  <xsl:choose>
    <xsl:when test="$d[@maxOccurs]">List[<xsl:value-of select="$t2" />]</xsl:when>    
    <xsl:otherwise><xsl:value-of select="$t2" /></xsl:otherwise>
  </xsl:choose> 
</xsl:function>

 

<xsl:variable name="FILE_HEADER" >

  Generate at <xsl:value-of  select="current-dateTime()" />

  by FyTok (rev: <xsl:value-of select="$FYTOK_REV"/>): builder/fy_imas.xsl

  from ITER Physics Data Model/IMAS DD, 
    version = <xsl:value-of select="/IDSs/version" />
    cocos   = <xsl:value-of select="/IDSs/cocos" /> 
  

</xsl:variable>

<xsl:variable name="util_dependence" select="()" />

<!-- Directory:  _imas -->
<xsl:template match="/">
  <!-- ids files -->
  <xsl:apply-templates select="./*" mode="INIT_FILE"/>
  <xsl:apply-templates select="./*" mode="IDS_FILE"/>

  <xsl:for-each select="xs:schema/xs:include">
    <xsl:choose>
      <xsl:when test="@schemaLocation='utilities/dd_support.xsd'"> 
        <xsl:apply-templates select="document(@schemaLocation)/*" mode="UTIL_FILE"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="document(@schemaLocation)/*" mode="IDS_FILES"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:for-each>
</xsl:template>


<!-- FILE:  __init__.py -->
<xsl:template match = "xs:schema" mode="INIT_FILE">
<xsl:result-document method="text" href="__init__.py">"""
  This package containes the _FyTok_ wrapper of IMAS/dd/ids
  <xsl:copy-of select="$FILE_HEADER" />
"""
__fy_rev__="<xsl:value-of select="$FYTOK_REV"/>"

__version__="<xsl:value-of select="/IDSs/version"/>"

__cocos__="<xsl:value-of select="/IDSs/cocos"/>"
  
  <xsl:for-each select="xs:include">
    <xsl:choose>
      <xsl:when test="@schemaLocation='utilities/dd_support.xsd'"></xsl:when>
      <xsl:otherwise>
  <xsl:variable name="ids_name" select="document(@schemaLocation)/*/xs:element/@name" />    
from .<xsl:value-of select="$ids_name"/>  import _T_<xsl:value-of select="$ids_name"/> 
      </xsl:otherwise>
    </xsl:choose>
  </xsl:for-each>

</xsl:result-document>
</xsl:template>

<!-- FILE:  _ids.py -->
<xsl:template match = "xs:schema" mode="IDS_FILE">

<xsl:message> DEBUG: create utilities.py </xsl:message>

<xsl:result-document method="text" href="_ids.py">"""
This package containes the base classes for  _FyTok_ _imas_wrapper
<xsl:copy-of select="$FILE_HEADER" />
"""
<xsl:value-of select="unparsed-text('fy_imas_ids.py')"/>
 
</xsl:result-document>
</xsl:template>

<!-- FILE:  utilities.py -->

<xsl:template match = "xs:schema" mode = "UTIL_FILE"> 

<xsl:result-document method="text" href='utilities.py'>"""
This package containes the _FyTok_ wrapper of IMAS/dd/utilities.py
<xsl:copy-of select="$FILE_HEADER" />
"""
import numpy as np
from spdm.data.Node import Node
from spdm.data.Function import Function 
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from enum import Enum

<xsl:apply-templates select="xs:complexType" mode="DEFINE"/>

</xsl:result-document>   
</xsl:template>


<!-- FILE:  {@name}.py -->
<xsl:template match = "xs:schema" mode = "IDS_FILES"> 
<xsl:variable name="filename" select="xs:element/@name"/>
<xsl:message> DEBUG: create <xsl:value-of select="$filename"/>.py </xsl:message>
<xsl:result-document method="text" href="{$filename}.py">"""
This package containes the _FyTok_ wrapper of IMAS/dd/ids
<xsl:copy-of select="$FILE_HEADER" />
"""
import numpy as np
from spdm.data.Node import Node
from spdm.data.Function import Function 
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from enum import Enum

from ._ids import IDS, Module, TimeSeries

<xsl:variable name="cls_list" select="for $k in xs:complexType/*/*/@type return if (not(xs:complexType[@name=$k]) and $k!='flt_type') then concat('_T_', $k) else ()"/>
<xsl:if test="count($cls_list) &gt; 0">
from .utilities import <xsl:value-of select="string-join(distinct-values($cls_list),',')"/>
</xsl:if>


<xsl:variable name="cls_list" select="for $k in xs:element/*/*/*/@type return if (not(xs:complexType[@name=$k]) and $k!='flt_type') then concat('_T_', $k) else ()"/>
<xsl:if test="count($cls_list) &gt; 0">
from .utilities import <xsl:value-of select="string-join(distinct-values($cls_list),',')"/>
</xsl:if>

<xsl:apply-templates select="xs:complexType" mode="DEFINE"/>

<xsl:apply-templates select="xs:element" mode="DEFINE_IDS"/>

</xsl:result-document>     
</xsl:template>



<!-- Declare element -->

<xsl:template match = "xs:appinfo">
<xsl:for-each select="./*"><xsl:value-of select="my:py_word(name(.))"/>="<xsl:value-of select="."/>",</xsl:for-each>
</xsl:template>

<xsl:template match = "xs:documentation"><xsl:value-of select="my:line-wrap(., $line-width, 7)"/></xsl:template>

<!-- Declare element -->

<xsl:template match = "xs:annotation">
  <xsl:apply-templates select="xs:documentation" />
  <xsl:text>&#xA;       </xsl:text>
  <xsl:for-each select="xs:appinfo/*">
        <xsl:text>&#xA;       </xsl:text><xsl:value-of select="my:py_word(name(.))"/>  : <xsl:value-of select="."/>
  </xsl:for-each>
  <xsl:text>&#xA;       </xsl:text>
</xsl:template>

<!-- Declare element -->
<xsl:template match = "xs:element[not(@ref)]" mode = "DECLARE">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_word(@name)"/> : <xsl:value-of select="my:type_hint(.)" /> =  sp_property(<xsl:apply-templates select="xs:annotation/xs:appinfo" />)
    """<xsl:value-of select="my:line-wrap(xs:annotation/xs:documentation, $line-width, 7)"/>"""
</xsl:template>

<xsl:template match = "xs:element[(@ref)]" mode = "DECLARE">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_word(@ref)"/> : <xsl:value-of select="my:type_hint(.)" /> =  sp_property(<xsl:apply-templates select="xs:annotation/xs:appinfo" />)
</xsl:template>

<!-- Define Class -->
<xsl:template match = "xs:complexType" mode = "DEFINE"> 

<xsl:variable name="base_class">
    <xsl:choose>      
      <xsl:when test="xs:sequence/xs:element[@name='time' and @type='flt_type']">TimeSeries</xsl:when>
      <xsl:when test="xs:sequence/xs:element[@name='code']" >Module</xsl:when>
      <xsl:otherwise>Dict[Node]</xsl:otherwise>
  </xsl:choose>
</xsl:variable>

class _T_<xsl:value-of select="@name" />(<xsl:value-of select="$base_class" />):
    """<xsl:apply-templates select="xs:annotation" />"""
    <xsl:apply-templates select="xs:sequence/xs:element[not(@name='time' or @name='code')]" mode="DECLARE" />
</xsl:template>

<!-- Define IDS -->
<xsl:template match = "xs:element" mode = "DEFINE_IDS"> 
class _T_<xsl:value-of select="@name" />(IDS):
    """<xsl:apply-templates select="xs:annotation" />"""
    <xsl:apply-templates select="xs:complexType/xs:sequence/xs:element[not(@ref)]" mode="DECLARE" />
</xsl:template>
</xsl:stylesheet>
