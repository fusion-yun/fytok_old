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

<xsl:param name="FYTOK_REV" select="'0.0.1'" />

<xsl:param name="DD_GIT_DESCRIBE" as="xs:string" />

<xsl:param name="BASE_DIR" as="xs:string" required='true' />

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

<!-- Directory:  _imas  -->
<xsl:template match="/*">  
  
  <xsl:call-template name="file_ids_py"/>

  <xsl:call-template name="file_init_py" />

  <!-- Scan for all constant identify ENUM -->
  <xsl:variable name="constants_list"   select="for $f in xs:include  return (document(concat($BASE_DIR,$f/@schemaLocation))//doc_identifier ) " />
  <xsl:variable name="constants_list"   select="for $f in $constants_list  return  if (starts-with($f,'utilities/')) then $f else () " />
  
  <xsl:call-template name="file_utilities_py">    
    <xsl:with-param name="constants_list" select="$constants_list" />
  </xsl:call-template>

  <xsl:for-each select="xs:include[@schemaLocation!='utilities/dd_support.xsd']">
      <xsl:apply-templates select="document(concat($BASE_DIR,./@schemaLocation))/*" mode="file_idsname_py" />   
  </xsl:for-each>
   
</xsl:template>


<!-- FILE:  __init__.py -->
<xsl:template name="file_init_py">
  <xsl:result-document method="text" href="__init__.py">"""
  This package containes the _FyTok_ wrapper of IMAS/dd/ids

  <xsl:copy-of select="$FILE_HEADER" />
"""
__fy_rev__  ="<xsl:value-of select="$FYTOK_REV"/>"
__version__ ="<xsl:value-of select="/IDSs/version"/>"
__cocos__   ="<xsl:value-of select="/IDSs/cocos"/>"
        
    <xsl:for-each select="xs:include[@schemaLocation!='utilities/dd_support.xsd']">
          <xsl:variable name="ids_name" select="document(concat($BASE_DIR,@schemaLocation))/*/xs:element/@name" />    
from .<xsl:value-of select="$ids_name"/>  import _T_<xsl:value-of select="$ids_name"/> 
    </xsl:for-each>

  </xsl:result-document>
</xsl:template>

<!-- FILE:  _ids.py -->
<xsl:template name="file_ids_py">
  <xsl:result-document method="text" href="_ids.py">""" 
    This package containes the base classes for  _FyTok_ _imas_wrapper <xsl:copy-of select="$FILE_HEADER" />
"""
<xsl:text>&#xA;</xsl:text>
<xsl:value-of select="unparsed-text('fy_imas_ids.py')"/>    
  </xsl:result-document>
</xsl:template>

<!-- FILE:  utilities.py -->
<xsl:template name="file_utilities_py">
  <xsl:param name="constants_list"/>
  
  <xsl:variable name="root" select="document(concat($BASE_DIR,'utilities/dd_support.xsd'))/*"/>

  <xsl:result-document method="text" href='utilities.py'>""" 
  This module containes the _FyTok_ wrapper of IMAS/dd/utilities.py 
  <xsl:copy-of select="$FILE_HEADER" />
"""
import numpy as np
from spdm.data.Node import Node
from spdm.data.Function import Function 
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from enum import Enum


    <xsl:for-each select="$constants_list"> 
      <xsl:apply-templates  select = "document(concat($BASE_DIR, .))/constants" mode = "CONSTANTS_IDENTIFY" /> 
    </xsl:for-each>

    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=0]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=1]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=2]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=3]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=4]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=5]" mode="DEFINE"/>
    <xsl:apply-templates select="$root/xs:complexType[my:dep_level(.,$root)=6]" mode="DEFINE"/>

    <xsl:apply-templates select="$root/xs:element" mode="DEFINE"/>

  </xsl:result-document>   
</xsl:template>

<!-- FILE:  {@name}.py -->
<xsl:template match = "xs:schema" mode = "file_idsname_py"> 
  <xsl:variable name="filename" select="xs:element/@name"/>
  <!-- <xsl:message> DEBUG: create <xsl:value-of select="$filename"/>.py </xsl:message> -->
  <xsl:result-document method="text" href="{$filename}.py"  >"""
  This module containes the _FyTok_ wrapper of IMAS/dd/<xsl:value-of select="xs:element/@name" />  
  <xsl:copy-of select="$FILE_HEADER" /> 
"""
import numpy as np
from spdm.data.Node import Node
from spdm.data.Function import Function 
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from enum import Enum

from ._ids import IDS, Module, TimeSlice

    <xsl:variable name="cls_list" select="for $k in //@type return if (not(xs:complexType[@name=$k]) and $k!='flt_type'  and $k!='flt_1d_type') then concat('_T_', $k) else ()"/>
    <xsl:variable name="cls_list" select="distinct-values($cls_list)"/>
    <xsl:if test="count($cls_list) &gt; 0">
from .utilities import <xsl:value-of select="string-join(distinct-values($cls_list),',')"/>
    </xsl:if>


    <xsl:variable name="cls_list1" select="for $k in //doc_identifier return if (starts-with($k,'utilities/')) then   $k  else ()"/>
    <xsl:for-each select="distinct-values($cls_list1)">
from .utilities import _E_<xsl:value-of select = "document(concat($BASE_DIR, .))/constants/@name"  /> 
    </xsl:for-each>

    <xsl:text>&#xA;    </xsl:text>

    <xsl:variable name="cls_list" select="for $k in //doc_identifier return if (not(starts-with($k,'utilities/'))) then   $k  else ()"/>
    <xsl:for-each select="distinct-values($cls_list)">
      <xsl:apply-templates  select = "document(concat($BASE_DIR, .))/constants" mode = "CONSTANTS_IDENTIFY" /> 
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

<!-- Declare element ######################################################################################### -->


<xsl:template match = "xs:documentation"><xsl:value-of select="my:line-wrap(., $line-width, 7)"/></xsl:template>

<xsl:template match = "xs:annotation">
  <xsl:apply-templates select="xs:documentation" />  
  <xsl:apply-templates select="xs:appinfo/*" />  
</xsl:template>

<xsl:template match="xs:sequence" mode="property_list">
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
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@name)"/> : _E_<xsl:value-of select = "document(concat($BASE_DIR, xs:annotation/xs:appinfo/doc_identifier))/constants/@name"  />  =  sp_property()
<xsl:text>    </xsl:text>"""<xsl:value-of select="my:line-wrap(xs:annotation/xs:documentation, $line-width, 7)"/>"""
      </xsl:when>      
    </xsl:choose>
  </xsl:for-each>  
</xsl:template>


<xsl:template match = "xs:complexType" mode = "DEFINE"> 
  <xsl:variable name="base_class">
      <xsl:choose>      
        <!-- <xsl:when test="xs:sequence/xs:element[@name='time' and @type='flt_type']">TimeSlice</xsl:when> -->
        <xsl:when test="xs:sequence/xs:element[@name='code']" >Module</xsl:when>
        <xsl:otherwise>Dict[Node]</xsl:otherwise>
    </xsl:choose>
  </xsl:variable>

class _T_<xsl:value-of select="@name" />(<xsl:value-of select="$base_class" />):
<xsl:text>    </xsl:text>"""<xsl:apply-templates select="xs:annotation" />"""
<xsl:apply-templates select="xs:sequence" mode="property_list" />
  
</xsl:template>

<xsl:template match = "constants" mode = "CONSTANTS_IDENTIFY"> 
class _E_<xsl:value-of select="@name"/>(Enum):
<xsl:text>    </xsl:text>"""<xsl:value-of select="my:line-wrap(header, $line-width, 7)"/>
     xpath: <xsl:value-of select="dd_instance/@xpath"/>
    """
  <xsl:for-each select="int">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_keyword(@name)"/> = <xsl:value-of select="."/> 
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@description, $line-width, 7)"/>"""
  </xsl:for-each>
</xsl:template>

<xsl:template match = "xs:element" mode = "DEFINE"> 
class _T_<xsl:value-of select="@name" />(Dict[Node]):
<xsl:text>    </xsl:text>"""<xsl:apply-templates select="xs:annotation" />"""
  
  <xsl:apply-templates select="xs:complexType/xs:sequence" mode="property_list" />
</xsl:template>

<xsl:template match = "xs:element" mode = "DEFINE_ELEMENT_AS_IDS"> 
class _T_<xsl:value-of select="@name" />(IDS):
<xsl:text>    </xsl:text>"""<xsl:apply-templates select="xs:annotation" />"""
  <xsl:apply-templates select="xs:complexType/xs:sequence" mode="property_list" />
</xsl:template>

</xsl:stylesheet>

