<?xml version="1.0" encoding="UTF-8"?>
<!--  
  Generate Python class (FyTok IDS) from IDSDef.xml file 
  
  copyright:
     @ASIPP, 2023,

  authors:
     Zhi YU, @ASIPP

  changes:
    2023-04-22: 0.0.1, ZY, initial version
    2023-04-23:        ZY, add function my:match_util,my:list_util,my:dep_level,my:to-camel-case,my:line-wrap
    2023-04-24:        ZY, test run ok

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

<xsl:param name="IDS_LIST" select="()" />

<xsl:param name="IMAS_DD_PATH" select="'/fuyun/software/data-dictionary/3.38.1/dd_3.38.1/include/'"/>

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

<xsl:variable name="FILE_HEADER" >

Generate at <xsl:value-of  select="current-dateTime()" />

  by FyTok (rev: <xsl:value-of select="$FYTOK_REV"/>): builder/fy_ids.xsl

  from ITER Physics Data Model/IMAS DD, version = <xsl:value-of select="/IDSs/version" />, cocos   = <xsl:value-of select="/IDSs/cocos" /> 
  

</xsl:variable>

<xsl:variable name="util_dependence" select="()" />

<!-- Directory:  _imas -->
<xsl:template match="/IDSs">
<xsl:apply-templates select="." mode="INIT_FILE"/>   
<xsl:apply-templates select="." mode="IDS_FILE"/>   
<xsl:apply-templates select="utilities" mode="FILE"/>   
<!-- <xsl:choose>
<xsl:when test="not(empty($IDS_LIST))">
<xsl:for-each select="IDS[not(empty(index-of($IDS_LIST,@name)))]">
  <xsl:value-of select="my:py_word(@name)"/><xsl:text>&#xA;</xsl:text> 
  <xsl:apply-templates select="." mode="FILE"/>
</xsl:for-each>
</xsl:when>
<xsl:otherwise> -->
<xsl:for-each select="IDS">
  <xsl:value-of select="my:py_word(@name)"/><xsl:text>&#xA;</xsl:text> 
  <xsl:apply-templates select="." mode="FILE"/>
</xsl:for-each>
<!-- </xsl:otherwise>
</xsl:choose> -->
</xsl:template>

<!-- FILE:  __init__.py -->
<xsl:template match = "IDSs" mode="INIT_FILE">
<xsl:result-document method="text" href="__init__.py">"""
This package containes the _FyTok_ wrapper of IMAS/dd/ids
<xsl:copy-of select="$FILE_HEADER" />
"""
__fy_rev__="<xsl:value-of select="$FYTOK_REV"/>"

__version__="<xsl:value-of select="/IDSs/version"/>"

__cocos__="<xsl:value-of select="/IDSs/cocos"/>"

from .utilities import _T_ids_properties, _T_code

<!-- <xsl:choose>
<xsl:when test="not(empty($IDS_LIST))">
<xsl:for-each select="IDS[not(empty(index-of($IDS_LIST,@name)))]">
from .<xsl:value-of select="my:py_word(@name)"/>  import _T_<xsl:value-of select="my:py_word(@name)"/>
</xsl:for-each>
</xsl:when>
<xsl:otherwise> -->
<xsl:for-each select="IDS">
from .<xsl:value-of select="my:py_word(@name)"/>  import _T_<xsl:value-of select="my:py_word(@name)"/>
</xsl:for-each>
<!-- </xsl:otherwise>
</xsl:choose> -->

</xsl:result-document>
</xsl:template>
 
<!-- FILE:  ids.py -->
<xsl:template match = "IDSs" mode="IDS_FILE">
<xsl:result-document method="text" href="ids.py">"""
This package containes the base classes for  _FyTok_ _imas_wrapper
<xsl:copy-of select="$FILE_HEADER" />
"""
<xsl:value-of select="unparsed-text('ids.py')"/>
 

</xsl:result-document>
</xsl:template>
 
<!-- FILE:  utilities.py -->
<xsl:template match="IDSs/utilities" mode="FILE" > 
<xsl:result-document method="text" href="utilities.py">"""
This module contains _FyTok_ wrappers of data structures defined in IMAS/dd_<xsl:value-of select="/IDSs/version"/>/utilities.
<xsl:copy-of select="$FILE_HEADER" />
"""
<xsl:variable name="cls_list" select="distinct-values(my:list_util(./field))"/>

import numpy as np
from spdm.data.Node import Node
from spdm.data.Function import Function
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from enum import IntFlag

<xsl:for-each select="field[my:dep_level(.)=0]">
<xsl:apply-templates select="." mode="DEFINE_UTIL"/>
</xsl:for-each>

<xsl:for-each select="field[my:dep_level(.)=1]">
<xsl:apply-templates select="." mode="DEFINE_UTIL"/>
</xsl:for-each>


<xsl:for-each select="field[my:dep_level(.)=2]">
<xsl:apply-templates select="." mode="DEFINE_UTIL"/>
</xsl:for-each>

<xsl:for-each select="field[my:dep_level(.)=3]">
<xsl:apply-templates select="." mode="DEFINE_UTIL"/>
</xsl:for-each>

<xsl:for-each select="field[my:dep_level(.)=4]">
<xsl:apply-templates select="." mode="DEFINE_UTIL"/>
</xsl:for-each>

</xsl:result-document>
</xsl:template>


<!-- FILE:  {IDS/@name}.py -->

<xsl:template match = "IDSs/IDS" mode="FILE">   
<xsl:result-document method="text" href="{@name}.py">"""
This module contains the _FyTok_ wrapper of IMAS/dd/<xsl:value-of select="my:py_word(@name)"/>
<xsl:copy-of select="$FILE_HEADER" />
"""
<xsl:variable name="cls_list" select="for $sub_field in field[(@name != 'code' and @name!='ids_properties')] return my:match_util($sub_field)"/>
<xsl:variable name="util_defined" select="distinct-values($cls_list)"/>
import numpy as np
from spdm.data.Node import Node
from spdm.data.Function import Function
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from enum import IntFlag

from .ids import _T_ids, _T_module
<xsl:choose>
<xsl:when test="(empty($util_defined))"/>
<xsl:otherwise>from .utilities import <xsl:value-of select="string-join(for $item in $util_defined return concat('_T_', $item), ',')"/></xsl:otherwise>
</xsl:choose>

<xsl:apply-templates select="field[(@data_type='structure' or @data_type='struct_array')]" mode = "DEFINE"/>  

class _T_<xsl:value-of select="my:py_word(@name)"/>(_T_ids):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""   
    
    _IDS = "<xsl:value-of select="my:py_word(@name)"/>" 

    <xsl:for-each select="field[@name!='ids_properties' and @name!='code' and @name!='time']" >
<xsl:text>&#xA;    </xsl:text><xsl:apply-templates select="." mode = "DECLARE"/>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""   
    </xsl:for-each>
</xsl:result-document>     
</xsl:template>

<!-- Declare field -->
<xsl:template match="field[@data_type='STR_0D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:str        = sp_property(type="<xsl:value-of select="@type"/>")   </xsl:template>
<xsl:template match="field[@data_type='STR_1D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:List[str]  = sp_property(type="<xsl:value-of select="@type"/>")   </xsl:template>
<xsl:template match="field[@data_type='str_type']"      mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:str        = sp_property(type="<xsl:value-of select="@type"/>")   </xsl:template>
<xsl:template match="field[@data_type='str_1d_type']"   mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:List[str]  = sp_property(type="<xsl:value-of select="@type"/>")   </xsl:template> 

<xsl:template match="field[@data_type='int_type']"      mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:int        = sp_property(type="<xsl:value-of select="@type"/>")   </xsl:template>    
<xsl:template match="field[@data_type='INT_0D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:int        = sp_property(type="<xsl:value-of select="@type"/>") </xsl:template>    

<xsl:template match="field[@data_type='INT_1D' or @data_type='int_1d_type']" mode = "DECLARE">  
<xsl:choose>
<xsl:when test="@type='constant' or @type='static'"><xsl:value-of select="my:py_word(@name)"/>:List[int]  = sp_property(type="<xsl:value-of select="@type"/>") </xsl:when>   
<xsl:otherwise>  <xsl:value-of select="my:py_word(@name)"/>:Function  = sp_property(type="<xsl:value-of select="@type"/>", coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:otherwise>   
</xsl:choose>
</xsl:template>   

<xsl:template match="field[@data_type='INT_2D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", ndims=2, data_type=int ) </xsl:template>   
<xsl:template match="field[@data_type='INT_3D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", ndims=3, data_type=int ) </xsl:template>   

<xsl:template match="field[@data_type='flt_type']"      mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:float      = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:template> 
<xsl:template match="field[@data_type='FLT_0D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:float      = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:template> 


<xsl:template match="field[@data_type='FLT_1D' or @data_type='flt_1d_type']" mode = "DECLARE">  
<xsl:choose>
<xsl:when test="@coordinate1='1...N'"><xsl:value-of select="my:py_word(@name)"/>:np.ndarray = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=float)</xsl:when>
<xsl:otherwise><xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />")</xsl:otherwise>
</xsl:choose>
</xsl:template>   

<xsl:template match="field[@data_type='FLT_2D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=2, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />") </xsl:template>   
<xsl:template match="field[@data_type='FLT_3D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=3, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />") </xsl:template>   
<xsl:template match="field[@data_type='FLT_4D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=4, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />") </xsl:template>   
<xsl:template match="field[@data_type='FLT_5D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=5, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />", coordinate5="<xsl:value-of select="@coordinate5" />") </xsl:template>   
<xsl:template match="field[@data_type='FLT_6D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=6, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />", coordinate5="<xsl:value-of select="@coordinate5" />", coordinate6="<xsl:value-of select="@coordinate6" />") </xsl:template>   
<xsl:template match="field[@data_type='cpx_type']"      mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:complex    = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:template>   
<xsl:template match="field[@data_type='cplx_1d_type']"  mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=complex, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:template>   
<xsl:template match="field[@data_type='CPX_0D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:complex    = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:template>   
<xsl:template match="field[@data_type='CPX_1D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=complex, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:template>   
<xsl:template match="field[@data_type='CPX_2D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=2, data_type=complex) </xsl:template>   
<xsl:template match="field[@data_type='CPX_3D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=3, data_type=complex) </xsl:template>   
<xsl:template match="field[@data_type='CPX_4D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=4, data_type=complex) </xsl:template>   
<xsl:template match="field[@data_type='CPX_5D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=5, data_type=complex) </xsl:template>   
<xsl:template match="field[@data_type='CPX_6D']"        mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:Function = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=6, data_type=complex) </xsl:template>   

<xsl:template match="field[@data_type='structure'     and not(@doc_identifier)]" mode = "DECLARE">  <xsl:value-of select="my:py_word(@name)"/>:_T_<xsl:value-of select="@structure_reference"/>         = sp_property() </xsl:template>    

<xsl:template match="field[@data_type='struct_array'  and not(@doc_identifier)]" mode = "DECLARE">  
<xsl:choose>
<xsl:when test="@coordinate1='1...N'"><xsl:value-of select="my:py_word(@name)"/>:List[_T_<xsl:value-of select="@structure_reference"/>]   = sp_property() </xsl:when>
<xsl:otherwise><xsl:value-of select="my:py_word(@name)"/>:List[_T_<xsl:value-of select="@structure_reference"/>]   = sp_property(coordinate1="<xsl:value-of select="@coordinate1"/>") </xsl:otherwise>
</xsl:choose>
</xsl:template>

<!-- Declare field: enum -->

<xsl:template match = "field[@doc_identifier]" mode = "DECLARE"> 
<xsl:variable name="ext_doc" select="document(concat($IMAS_DD_PATH,@doc_identifier))/constants" />
<xsl:text>&#xA;    </xsl:text>
  <xsl:choose>
    <xsl:when test="@data_type='structure'"><xsl:value-of select="my:py_word(@name)"/>:_E_<xsl:value-of select="$ext_doc/[@name]"/>  = sp_property() </xsl:when>
    <xsl:when test="@data_type='struct_array' and @coordinate1='1...N'"><xsl:value-of select="my:py_word(@name)"/>:List[_E_<xsl:value-of select="$ext_doc/[@name]"/>]   = sp_property() </xsl:when>
    <xsl:when test="@data_type='struct_array' and @coordinate1!='1...N'"><xsl:value-of select="my:py_word(@name)"/>:List[_E_<xsl:value-of select="$ext_doc/[@name]"/>]   = sp_property(coordinate1="<xsl:value-of select="@coordinate1"/>") </xsl:when>
    <xsl:otherwise> # unknown data type <xsl:value-of select="my:py_word(@name)"/>:_E_<xsl:value-of select="my:py_word(@name)"/>  </xsl:otherwise>
  </xsl:choose>
</xsl:template>
 


 <!-- Define field: enum -->
<xsl:template match = "field[@doc_identifier]" mode = "DEFINE"> 
  <xsl:variable name="ext_doc" select="document(concat($IMAS_DD_PATH,@doc_identifier))/constants" />
class _E_<xsl:value-of select="$ext_doc/[@name]"/>(IntFlag):
    """<xsl:value-of select="my:line-wrap($ext_doc/header, $line-width, 7)"/>"""
  <xsl:for-each select="$ext_doc/int">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_word(@name)"/>:   <xsl:value-of select="."/>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@description, $line-width, 7)"/>"""
  </xsl:for-each>
</xsl:template>

<!-- Define field: dataclass -->
<xsl:template match = "field[not(@doc_identifier)]" mode = "DEFINE"> 
<xsl:variable name="structure_reference" select="@structure_reference"/>

<xsl:if test="not(/IDSs/utilities/field[(@data_type='structure' and @name=$structure_reference)])"> 

<xsl:apply-templates select="field[(@data_type='structure' or @data_type='struct_array')]" mode = "DEFINE"/>  

class _T_<xsl:value-of select="$structure_reference"/>(Dict[Node]<xsl:if test="field[@name='code']">, _T_module</xsl:if>):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""
    <xsl:if test="field[@name='code']">
    _registry = {}
    _plugin_prefix="fytok/plugins/<xsl:value-of select="$structure_reference"/>"
    </xsl:if>    

    <xsl:for-each select="field[substring(@name,string-length(@name)-string-length('_error_upper')+1)!='_error_upper'
      and substring(@name,string-length(@name)-string-length('_error_lower')+1)!='_error_lower'
      and substring(@name,string-length(@name)-string-length('_error_index')+1)!='_error_index'
      ]" >
<xsl:text>&#xA;    </xsl:text><xsl:apply-templates select="." mode = "DECLARE"/>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""   
    </xsl:for-each>


</xsl:if>
</xsl:template>


<xsl:template match = "field[@doc_identifier]" mode = "DEFINE_UTIL">     

<xsl:variable name="ext_doc" select="document(concat($IMAS_DD_PATH,@doc_identifier))/constants" />
class _E_<xsl:value-of select="$ext_doc/[@name]"/>(IntFlag):
    """<xsl:value-of select="my:line-wrap($ext_doc/header, $line-width, 7)"/>"""
  <xsl:for-each select="$ext_doc/int">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="my:py_word(@name)"/>:   <xsl:value-of select="."/>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@description, $line-width, 7)"/>"""
  </xsl:for-each>
    
 
</xsl:template>

<xsl:template match = "field[not(@doc_identifier)]" mode = "DEFINE_UTIL">     
<xsl:apply-templates select="field[(@data_type='structure' or @data_type='struct_array')]" mode = "DEFINE_UTIL"/>  

class _T_<xsl:value-of select="my:py_word(@name)"/>(Dict[Node]):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""

  <xsl:for-each select="field[substring(@name,string-length(@name)-string-length('_error_upper')+1)!='_error_upper'
      and substring(@name,string-length(@name)-string-length('_error_lower')+1)!='_error_lower'
      and substring(@name,string-length(@name)-string-length('_error_index')+1)!='_error_index'
      ]" >
<xsl:text>&#xA;    </xsl:text><xsl:apply-templates select="." mode = "DECLARE"/>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""   
  </xsl:for-each>
 
</xsl:template>

</xsl:stylesheet>
