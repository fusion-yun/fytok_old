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
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="no"/>
<!-- <xsl:strip-space elements="*"/> -->

<xsl:param name="FY_GIT_DESCRIBE" as="xs:string" />

<xsl:param name="DD_GIT_DESCRIBE" as="xs:string" />

<xsl:param name="DD_BASE_DIR" as="xs:string" required='true' />

<!-- <xsl:param name="FILE_HEADER_ANNOTATION" select=" N/A " /> -->

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
  <xsl:param name="node" as="element()*"/>
  <xsl:param name="root" as="element()*"/>
  <xsl:variable name="children" select="for $sub_node in $node/xs:sequence/xs:element[@type] return my:dep_level($root/xs:complexType[@name=$sub_node/@type],$root)"/>
  <xsl:choose>
    <xsl:when test="empty($children)"> <xsl:sequence select="0"/></xsl:when>
    <xsl:otherwise> <xsl:sequence select="1+max($children)"/> </xsl:otherwise>
  </xsl:choose>  
</xsl:function>

<xsl:function name="my:py_keyword">
  <xsl:param name="word"/>
  <xsl:variable name="keywords" select="'and,as,assert,break,class,continue,def,del,elif,else,except,False,finally,for,from,global,if,import,in,is,lambda,None,nonlocal,not,or,pass,Raise,True,Try,while,yield'"/>
  <xsl:variable name="is-keyword" select="contains(concat(',', $keywords, ','), concat(',', $word, ','))"/>
  <xsl:variable name="word-with-underscores" select="translate($word, ' /', '_')"/>
  <xsl:choose>
    <xsl:when test="$is-keyword">
      <xsl:value-of select="concat( $word-with-underscores,'_')"/>
    </xsl:when>
    <xsl:otherwise>
      <xsl:value-of select="$word-with-underscores"/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:function>

<xsl:function name="my:quote">
  <xsl:param name="str" />
  <xsl:choose>
    <xsl:when test="starts-with($str,'&quot;') and ends-with($str,'&quot;')"><xsl:value-of select="$str" /></xsl:when>
    <xsl:when test="starts-with($str,'&apos;') and ends-with($str,'&apos;')"><xsl:value-of select="$str" /></xsl:when>
    <xsl:otherwise><xsl:value-of select="concat('&quot;', $str, '&quot;')" /></xsl:otherwise>
  </xsl:choose>    
</xsl:function>


<xsl:variable name="FILE_HEADER_ANNOTATION" >
  Generate at <xsl:value-of  select="current-dateTime()" />
  by FyTok (rev: <xsl:value-of select="$FY_GIT_DESCRIBE"/>): builder/fy_imas_xsd.xsl
</xsl:variable>


<xsl:variable name="FILE_HEADER_COMMON_IMPORT" >
from enum import IntFlag
import numpy as np
from spdm.data.Node         import Node
from spdm.data.List         import List
from spdm.data.Dict         import Dict
from spdm.data.TimeSeries   import TimeSeriesAoS,TimeSlice
from spdm.data.Signal       import Signal 
from spdm.data.Profile      import Profile 
from spdm.data.sp_property  import sp_property

</xsl:variable>

<!-- Directory:  _imas  -->
<xsl:template match="/*">  

  <xsl:apply-templates select="xs:element[@name='physics_data_dictionary']" mode="file_init_py" />

  <!-- Scan for all constant identify ENUM -->
  <xsl:variable name="constants_list"   select="for $f in xs:include  return (document(concat($DD_BASE_DIR,$f/@schemaLocation))//doc_identifier ) " />
  <xsl:variable name="constants_list"   select="for $f in $constants_list  return  if (starts-with($f,'utilities/')) then $f else () " />
  
  <xsl:call-template name="file_utilities_py">    
    <xsl:with-param name="constants_list" select="$constants_list" />
  </xsl:call-template>

  <xsl:for-each select="xs:include[@schemaLocation!='utilities/dd_support.xsd']">
      <xsl:apply-templates select="document(concat($DD_BASE_DIR,./@schemaLocation))/*" mode="file_idsname_py" />   
  </xsl:for-each>
   
</xsl:template>


<!-- FILE:  __init__.py -->
<xsl:template match="xs:element[@name='physics_data_dictionary']" mode="file_init_py">
<xsl:result-document method="text" href="__init__.py">"""
  <xsl:value-of select="xs:annotation/xs:documentation"/>

  From IMAS/dd (<xsl:value-of select="$DD_GIT_DESCRIBE"/>)
  <xsl:copy-of select="$FILE_HEADER_ANNOTATION" />
"""
__fy_rev__  ="<xsl:value-of select="$FY_GIT_DESCRIBE"/>"
__version__ ="<xsl:value-of select="$DD_GIT_DESCRIBE"/>"
__cocos__   ="<xsl:value-of select="xs:annotation/xs:appinfo/cocos"/>"
        
<xsl:for-each select="xs:complexType/xs:sequence/xs:element">
from .<xsl:value-of select="@ref"/>  import _T_<xsl:value-of select="@ref"/> 
</xsl:for-each>

</xsl:result-document>
</xsl:template>

<!-- FILE:  utilities.py -->
<xsl:template name="file_utilities_py">
  <xsl:param name="constants_list"/>
  <xsl:variable name="root" select="document(concat($DD_BASE_DIR,'utilities/dd_support.xsd'))/*"/>
<xsl:result-document method="text" href='utilities.py'>""" 
    This module containes the _FyTok_ wrapper of IMAS/dd/utilities.py 

  <xsl:copy-of select="$FILE_HEADER_ANNOTATION" />
"""
    <xsl:copy-of select="$FILE_HEADER_COMMON_IMPORT" />


    <xsl:for-each select="$constants_list"> 
      <xsl:apply-templates  select = "document(concat($DD_BASE_DIR, .))/constants" mode = "CONSTANTS_IDENTIFY" /> 
    </xsl:for-each>

    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=0]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=1]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=2]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=3]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=4]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=5]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=6]" mode="DEFINE"/>

    <xsl:apply-templates select="$root/xs:element" mode="DEFINE"/>
    <xsl:value-of select="unparsed-text('fy_imas.py')"/>


  </xsl:result-document>   
</xsl:template>

<!-- FILE:  {@name}.py -->
<xsl:template match = "xs:schema" mode = "file_idsname_py"> 
  <xsl:variable name="filename" select="xs:element/@name"/>
  <!-- <xsl:message> DEBUG: create <xsl:value-of select="$filename"/>.py </xsl:message> -->
  <xsl:result-document method="text" href="{$filename}.py"  >"""
  This module containes the _FyTok_ wrapper of IMAS/dd/<xsl:value-of select="xs:element/@name" />  

  <xsl:copy-of select="$FILE_HEADER_ANNOTATION" /> 
"""
<xsl:copy-of select="$FILE_HEADER_COMMON_IMPORT" />

from .utilities import _T_IDS, _T_Module

    <xsl:variable name="cls_list" select="for $k in //@type return if (not(xs:complexType[@name=$k]) and $k!='flt_type'  and $k!='flt_1d_type') then concat('_T_', $k) else ()"/>
    <xsl:variable name="cls_list" select="distinct-values($cls_list)"/>
    <xsl:if test="count($cls_list) &gt; 0">
from .utilities import <xsl:value-of select="string-join(distinct-values($cls_list),',')"/>
    </xsl:if>


    <xsl:variable name="cls_list1" select="for $k in //doc_identifier return if (starts-with($k,'utilities/')) then   $k  else ()"/>
    <xsl:for-each select="distinct-values($cls_list1)">
from .utilities import _E_<xsl:value-of select = "document(concat($DD_BASE_DIR, .))/constants/@name"  /> 
    </xsl:for-each>

    <xsl:text>&#xA;    </xsl:text>

    <xsl:variable name="cls_list" select="for $k in //doc_identifier return if (not(starts-with($k,'utilities/'))) then   $k  else ()"/>
    <xsl:for-each select="distinct-values($cls_list)">
      <xsl:apply-templates  select = "document(concat($DD_BASE_DIR, .))/constants" mode = "CONSTANTS_IDENTIFY" /> 
    </xsl:for-each>

    <xsl:variable name="root" select="." />
    
    <xsl:apply-templates select="xs:complexType[my:dep_level(.,$root)=0]" mode="DEFINE"/>    
    <xsl:apply-templates select="xs:complexType[my:dep_level(.,$root)=1]" mode="DEFINE"/>
    <xsl:apply-templates select="xs:complexType[my:dep_level(.,$root)=2]" mode="DEFINE"/>
    <xsl:apply-templates select="xs:complexType[my:dep_level(.,$root)=3]" mode="DEFINE"/>
    <xsl:apply-templates select="xs:complexType[my:dep_level(.,$root)=4]" mode="DEFINE"/>
    <xsl:apply-templates select="xs:complexType[my:dep_level(.,$root)=5]" mode="DEFINE"/>
    <xsl:apply-templates select="xs:complexType[my:dep_level(.,$root)=6]" mode="DEFINE"/>

    <xsl:apply-templates select="xs:element" mode="DEFINE_ELEMENT_AS_IDS"/>

  </xsl:result-document>     
</xsl:template>

<!-- as_python_value 将 节点内容转换为 python dict/list/string
  - 如果节点是一个简单类型，则它将其转换为一个字符串。
  - 如果节点是一个复杂类型，则它将其转换为一个字典，其中包含所有子节点的值。
  - 如果有多个同名兄弟节点，则它将它们存储在一个列表中；否则，它将它们存储为单个值。   
-->

<xsl:template match="*" mode="as_python_kw">
  <xsl:if test="not(preceding-sibling::*[name() = name(current())])">
    <xsl:value-of select="my:quote(name())" /><xsl:text>: </xsl:text>
    <xsl:variable name="siblings" select="../child::*[name() = name(current())]" />
    <xsl:choose>
      <xsl:when test="count($siblings) > 1">
        <xsl:text>[</xsl:text>
        <xsl:apply-templates select="$siblings" mode="as_python_value"/>
        <xsl:text>]</xsl:text>
      </xsl:when>
      <xsl:otherwise><xsl:value-of select="my:quote($siblings)" /></xsl:otherwise>
    </xsl:choose>
    <xsl:if test="position() != last()">,</xsl:if>
  </xsl:if>
</xsl:template>

<xsl:template match="*" mode="as_python_value">
  <xsl:choose>
    <xsl:when test="./*">
      <xsl:text>{</xsl:text><xsl:apply-templates select="*" mode="as_python_kw" /><xsl:text>}</xsl:text>
    </xsl:when>
    <xsl:otherwise><xsl:value-of select="my:quote(.)" /></xsl:otherwise>
  </xsl:choose>
  <xsl:if test="position() != last()">,</xsl:if>
</xsl:template>

<!-- as_python_kwargs 将子节点转换为 key=value 形式，可用作 python 函数 kwargs 参数  
  - 如果有多个同名兄弟节点，则它将它们存储在一个列表中；否则，它将它们存储为单个值。
-->

<xsl:template match="*" mode="as_python_kwargs_one">  
  <xsl:if test="not(preceding-sibling::*[name() = name(current())])">
    <xsl:value-of select="my:py_keyword(name())" /><xsl:text>=</xsl:text>
    <xsl:variable name="siblings" select="../child::*[name() = name(current())]" />
    <xsl:choose>
      <xsl:when test="count($siblings) > 1"><xsl:text>[</xsl:text><xsl:apply-templates select="$siblings" mode="as_python_value"/><xsl:text>]</xsl:text></xsl:when>
      <xsl:otherwise><xsl:apply-templates select="$siblings"  mode="as_python_value" /> </xsl:otherwise>
    </xsl:choose>
    <xsl:if test="position() != last()">,</xsl:if>
  </xsl:if>  
</xsl:template>

<xsl:template match="*" mode="as_python_kwargs">
  <xsl:choose>
    <xsl:when test="./*">
      <xsl:apply-templates select="*" mode="as_python_kwargs_one" />
    </xsl:when>
    <xsl:otherwise></xsl:otherwise>
  </xsl:choose>
  
</xsl:template>

<!--  ######################################################################################### -->


<xsl:template match = "xs:documentation">"""<xsl:value-of select="my:line-wrap(., $line-width, 7)"/>"""</xsl:template>

<xsl:template match = "xs:annotation">"""
    <xsl:value-of select="my:line-wrap(xs:documentation, $line-width, 7)"/>  

    <xsl:apply-templates select="xs:appinfo/*" />
    """    
</xsl:template>

<!-- Declare element ######################################################################################### -->

<xsl:variable name="type_map">
    <entry key='STR_0D'       >str</entry>
    <entry key='str_type'     >str</entry> 
    <entry key='STR_1D'       >List[str]</entry>
    <entry key='str_1d_type'  >List[str]</entry>
    <entry key='INT_0D'       >int</entry>
    <entry key='int_type'     >int</entry>
    <entry key='INT_1D'       >Profile[int]</entry>
    <entry key='int_1d_type'  >Profile[int]</entry>
    <entry key='INT_2D'       >Profile[int]</entry>
    <entry key='INT_3D'       >Profile[int]</entry>
    <entry key='INT_4D'       >Profile[int]</entry>
    <entry key='INT_5D'       >Profile[int]</entry>
    <entry key='INT_6D'       >Profile[int]</entry>
    <entry key='FLT_0D'       >float</entry>
    <entry key='flt_type'     >float</entry>
    <entry key='FLT_1D'       >Profile[float]</entry>
    <entry key='flt_1d_type'  >Profile[float]</entry>
    <entry key='FLT_2D'       >Profile[float]</entry>
    <entry key='FLT_3D'       >Profile[float]</entry>
    <entry key='FLT_4D'       >Profile[float]</entry>
    <entry key='FLT_5D'       >Profile[float]</entry>
    <entry key='FLT_6D'       >Profile[float]</entry>
    <entry key='cpx_type'     >complex</entry>
    <entry key='cplx_1d_type' >Profile[complex]</entry>
    <entry key='CPX_0D'       >Profile[complex]</entry>
    <entry key='CPX_1D'       >Profile[complex]</entry>
    <entry key='CPX_2D'       >Profile[complex]</entry>
    <entry key='CPX_3D'       >Profile[complex]</entry>
    <entry key='CPX_4D'       >Profile[complex]</entry>
    <entry key='CPX_5D'       >Profile[complex]</entry>
    <entry key='CPX_6D'       >Profile[complex]</entry>
    
    <entry key='signal_flt_1d'>Signal[float]</entry>
    <entry key='signal_flt_2d'>Signal[float]</entry>
    <entry key='signal_flt_3d'>Signal[float]</entry>
    <entry key='signal_flt_4d'>Signal[float]</entry>
    <entry key='signal_flt_5d'>Signal[float]</entry>
    <entry key='signal_flt_6d'>Signal[float]</entry>

    <entry key='signal_int_1d'>Signal[int]</entry>
    <entry key='signal_int_2d'>Signal[int]</entry>
    <entry key='signal_int_3d'>Signal[int]</entry>
    <entry key='signal_int_4d'>Signal[int]</entry>
    <entry key='signal_int_5d'>Signal[int]</entry>
    <entry key='signal_int_6d'>Signal[int]</entry>


</xsl:variable>

<xsl:template match="xs:element[@name  or  @ref]" mode="DECLARE">
<xsl:variable name="prop_name">
  <xsl:choose>
    <xsl:when test="@ref"><xsl:value-of select="my:py_keyword(@ref)"/></xsl:when>
    <xsl:otherwise><xsl:value-of select="my:py_keyword(@name)"/></xsl:otherwise>
  </xsl:choose>
</xsl:variable>
<xsl:variable name="type_hint">
  <xsl:choose>
    <xsl:when test="@ref"><xsl:value-of select="@ref"/></xsl:when>
    <xsl:when test="@type"><xsl:value-of select="@type"/></xsl:when>
    <xsl:otherwise><xsl:value-of select="xs:complexType/xs:group/@ref"/></xsl:otherwise>
  </xsl:choose>
</xsl:variable>
<xsl:variable name="type_hint">
  <xsl:choose>
    <xsl:when test="($type_hint='INT_1D' or $type_hint='int_1d_type') and normalize-space(xs:annotation/xs:appinfo/coordinate1)='1...N' ">List[int]</xsl:when>          
    <xsl:when test="($type_hint='FLT_1D' or $type_hint='flt_1d_type') and normalize-space(xs:annotation/xs:appinfo/coordinate1)='1...N' ">np.ndarray</xsl:when>          
    <xsl:when test="$type_map/entry[@key=$type_hint]"><xsl:value-of select="$type_map/entry[@key=$type_hint]"/></xsl:when>          
    <xsl:when test="xs:annotation/xs:appinfo/doc_identifier">_E_<xsl:value-of select = "document(concat($DD_BASE_DIR, xs:annotation/xs:appinfo/doc_identifier))/constants/@name"/></xsl:when>
    <xsl:otherwise>_T_<xsl:value-of select="$type_hint"/> </xsl:otherwise>   
  </xsl:choose>
</xsl:variable>

<xsl:variable name="type_hint">
  <xsl:choose>
    <xsl:when test="@maxOccurs">
      <xsl:choose>
        <xsl:when test="ends-with(xs:annotation/xs:appinfo/coordinate1,'time')">TimeSeriesAoS[<xsl:value-of select="$type_hint" />]</xsl:when>      
        <xsl:otherwise>List[<xsl:value-of select="$type_hint" />]</xsl:otherwise>               
      </xsl:choose>
    </xsl:when>   
    <xsl:otherwise><xsl:value-of select="$type_hint"/> </xsl:otherwise>   
  </xsl:choose>
</xsl:variable>
<xsl:if test="not(xs:annotation/xs:appinfo/lifecycle_status)  or  xs:annotation/xs:appinfo/lifecycle_status!='obsolescent'" >
<xsl:text>&#xA;&#xA;    </xsl:text><xsl:value-of select="$prop_name"/>  :<xsl:value-of select="$type_hint" /> =  sp_property(<xsl:apply-templates select="xs:annotation/xs:appinfo"  mode="as_python_kwargs"/>)
<xsl:text>    </xsl:text><xsl:apply-templates select="xs:annotation/xs:documentation"/>
</xsl:if>
</xsl:template>
 

<!-- <xsl:template match="xs:sequence" mode="property_list">
  <xsl:for-each select="xs:element[@name!='code' and @name!='time' and @name!='ids_properties' ]">
    <xsl:choose>
      <xsl:when test = "@ref" >
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@ref)"/>   : <xsl:value-of select="my:type_hint(.)" /> =  sp_property(<xsl:apply-templates select="xs:annotation/xs:appinfo"  mode="as_python_kwargs"/>)
      </xsl:when>
      <xsl:when test= "not(@ref) and not(xs:annotation/xs:appinfo/doc_identifier)">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@name)"/>  : <xsl:value-of select="my:type_hint(.)" /> =  sp_property(<xsl:apply-templates select="xs:annotation/xs:appinfo"  mode="as_python_kwargs"/>)
<xsl:text>    </xsl:text>"""<xsl:value-of select="my:line-wrap(xs:annotation/xs:documentation, $line-width, 7)"/>"""
      </xsl:when>
      <xsl:when test = "not(@ref) and (xs:annotation/xs:appinfo/doc_identifier)">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@name)"/> : _E_<xsl:value-of select = "document(concat($DD_BASE_DIR, xs:annotation/xs:appinfo/doc_identifier))/constants/@name"  />  =  sp_property()
<xsl:text>    </xsl:text>"""<xsl:value-of select="my:line-wrap(xs:annotation/xs:documentation, $line-width, 7)"/>"""
      </xsl:when>      
    </xsl:choose>
  </xsl:for-each>  
</xsl:template> -->


<xsl:template match = "xs:complexType" mode = "DEFINE"> 
<xsl:choose>      
<xsl:when test="xs:sequence/xs:element[@name='code']" >
<xsl:text>&#xA;&#xA;</xsl:text>class _T_<xsl:value-of select="@name" />(_T_Module):
<xsl:text>    </xsl:text><xsl:apply-templates select="xs:annotation" />
<xsl:apply-templates select="xs:sequence/xs:element[@name!='code']" mode="DECLARE" />
</xsl:when>
<xsl:when test="xs:sequence/xs:element[@name='time'] and xs:sequence/xs:element[@name='time'][@type='flt_type'] " >
<xsl:text>&#xA;&#xA;</xsl:text>class _T_<xsl:value-of select="@name" />(TimeSlice):
<xsl:text>    </xsl:text><xsl:apply-templates select="xs:annotation" />
<xsl:apply-templates select="xs:sequence/xs:element[@name!='time']" mode="DECLARE" />
</xsl:when>
<xsl:otherwise>
<xsl:text>&#xA;&#xA;</xsl:text>class _T_<xsl:value-of select="@name" />(Dict[Node]):
<xsl:text>    </xsl:text><xsl:apply-templates select="xs:annotation" />
<xsl:apply-templates select="xs:sequence/xs:element" mode="DECLARE" />
</xsl:otherwise>
</xsl:choose>
</xsl:template>

<xsl:template match = "constants[@identifier='yes']" mode = "CONSTANTS_IDENTIFY"> 
<xsl:text>&#xA;&#xA;</xsl:text>class _E_<xsl:value-of select="@name"/>(IntFlag):
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(header, $line-width, 7)"/>
     xpath: <xsl:value-of select="dd_instance/@xpath"/>
    """
  <xsl:for-each select="int">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@name)"/> = <xsl:value-of select="."/> 
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@description, $line-width, 7)"/>"""
  </xsl:for-each>
</xsl:template>

<xsl:template match = "xs:element" mode = "DEFINE"> 
<xsl:text>&#xA;</xsl:text>class _T_<xsl:value-of select="@name" />(Dict[Node]):
<xsl:text>&#xA;    </xsl:text><xsl:apply-templates select="xs:annotation" />
  
<xsl:apply-templates select="xs:complexType/xs:sequence/xs:element" mode="DECLARE" />
</xsl:template>

<xsl:template match = "xs:element" mode = "DEFINE_ELEMENT_AS_IDS"> 

from .utilities import  _T_ids_properties,_T_code,_T_time
<xsl:text>&#xA;</xsl:text>class _T_<xsl:value-of select="@name" />(_T_IDS):
<xsl:text>&#xA;    </xsl:text><xsl:apply-templates select="xs:annotation" />

<xsl:apply-templates select="xs:complexType/xs:sequence/xs:element" mode="DECLARE" />
</xsl:template>

</xsl:stylesheet>

