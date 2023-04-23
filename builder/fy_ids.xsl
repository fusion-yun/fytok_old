<?xml version="1.0" encoding="UTF-8"?>
<!--  
  Generate Python class (FyTok IDS) from IDSDef.xml file 
  
  copyright:
     @ASIPP, 2023,

  authors:
     Zhi YU, @ASIPP

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

<xsl:param name="IDS_LIST" select="('distributions')" />

<xsl:param name="IDS_FIELDS" select="('ids_properties','code')" />

<xsl:param name="DD_PATH" select="'/fuyun/software/data-dictionary/3.38.1/dd_3.38.1/include/'"/>

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
  <xsl:variable name="wrapped-text" select="replace(concat(normalize-space($text),' '), concat('(.{0,', $line-length, '}) '), concat('$1&#10;', $spaces))" />
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

<xsl:variable name="FILE_HEADER" >

Generate by fytok/builder/fy_ids.xsl (rev="<xsl:value-of select="$FYTOK_REV"/>")
  from ITER Physics Data Model/IMAS DD,  version=<xsl:value-of select="/IDSs/version" />, cocos = <xsl:value-of select="/IDSs/cocos" /> 
  at <xsl:value-of  select="current-dateTime()" />

</xsl:variable>

<xsl:variable name="util_dependence" select="()" />

<!-- Directory:  _imas -->
<xsl:template match="/IDSs">

<xsl:apply-templates select="." mode="INIT_FILE"/>   
<xsl:apply-templates select="." mode="IDS_FILE"/>   

<xsl:for-each select="IDS[contains($IDS_LIST,@name)]">

<xsl:value-of select="@name"/><xsl:text>&#xA;</xsl:text> 

<xsl:apply-templates select="." mode="FILE"/>   

</xsl:for-each>

<xsl:apply-templates select="utilities" mode="FILE"/>   

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

<xsl:for-each select="IDS[contains($IDS_LIST,@name)]">
from .<xsl:value-of select="@name"/>  import _T_<xsl:value-of select="@name"/>
</xsl:for-each>
</xsl:result-document>
</xsl:template>
 
<!-- FILE:  ids.py -->
<xsl:template match = "IDSs" mode="IDS_FILE">
<xsl:result-document method="text" href="ids.py">"""
This package containes the _FyTok_ base class _T_ids
<xsl:copy-of select="$FILE_HEADER" />
"""
import numpy as np
from spdm.data.Node import Node
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property

from .utilities import _T_ids_properties, _T_code

class _T_ids(Dict[Node]):
    """ Base class of IDS """

    ids_properties:_T_ids_properties  = sp_property()
    """Interface Data Structure properties. This element identifies the node above as an IDS"""
    
    code:_T_code  = sp_property()
    """Generic decription of the code-specific parameters for the code that has produced this IDS"""

    time:np.ndarray = sp_property(type="dynamic", units="s", ndims=1, data_type=float, coordinate1="1...N") 
    """Generic time"""
</xsl:result-document>
</xsl:template>
 
<!-- FILE:  utilities.py -->
<xsl:template match="IDSs/utilities" mode="FILE" > 
<xsl:result-document method="text" href="utilities.py">"""
This module contains _FyTok_ wrappers of data structures defined in IMAS/dd_<xsl:value-of select="/IDSs/version"/>/utilities.
<xsl:copy-of select="$FILE_HEADER" />
"""
<xsl:variable name="cls_list" select="distinct-values(my:list_util(./field))"/>
from spdm.data.Node import Node
from spdm.data.ndFunction import ndFunction
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
This module contains the _FyTok_ wrapper of IMAS/dd/<xsl:value-of select="@name"/>
<xsl:copy-of select="$FILE_HEADER" />
"""
<xsl:variable name="cls_list" select="for $sub_field in field[(@name != 'code' and @name!='ids_properties')] return my:match_util($sub_field)"/>
<xsl:variable name="util_defined" select="distinct-values($cls_list)"/>

from spdm.data.Node import Node
from spdm.data.ndFunction import ndFunction
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property
from enum import IntFlag

from .ids import _T_ids
from .utilities import <xsl:value-of select="string-join(for $item in $util_defined return concat('_T_', $item), ',')"/>


<xsl:apply-templates select="field[(@data_type='structure' or @data_type='struct_array')]" mode = "DEFINE"/>  

class _T_<xsl:value-of select="@name"/>(_T_ids):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""   

    _IDS = "<xsl:value-of select="@name"/>" 

    <xsl:for-each select="field" >
      <xsl:if test="@name!='ids_properties' and @name!='code' and @name!='time'">
        <xsl:apply-templates select="." mode = "DECLARE"/>   
      </xsl:if>
    </xsl:for-each>
</xsl:result-document>     
</xsl:template>

<!-- Declare field -->

<xsl:template match = "field[not(@doc_identifier)]" mode = "DECLARE"> 
<xsl:text>&#xA;    </xsl:text>
  <xsl:choose>
    <xsl:when test="@data_type='str_type'    or @data_type='STR_0D'">  <xsl:value-of select="@name"/>:str       = sp_property(type="<xsl:value-of select="@type"/>")    </xsl:when>
    <xsl:when test="@data_type='str_1d_type' or @data_type='STR_1D'">  <xsl:value-of select="@name"/>:List[str] = sp_property(type="<xsl:value-of select="@type"/>" )    </xsl:when>   

    <xsl:when test="@data_type='int_type'">       <xsl:value-of select="@name"/>:int        = sp_property(type="<xsl:value-of select="@type"/>") </xsl:when>    
    <xsl:when test="@data_type='int_1d_type'">    <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", ndims=1, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='INT_0D'">         <xsl:value-of select="@name"/>:int        = sp_property(type="<xsl:value-of select="@type"/>") </xsl:when>    
    <xsl:when test="@data_type='INT_1D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", ndims=1, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='INT_2D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", ndims=2, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='INT_3D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", ndims=3, data_type=int ) </xsl:when>   
    <xsl:when test="@data_type='flt_type'">       <xsl:value-of select="@name"/>:float      = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when> 
    <xsl:when test="@data_type='flt_1d_type'">    <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>       
    <xsl:when test="@data_type='FLT_0D'">         <xsl:value-of select="@name"/>:float      = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when> 
    <xsl:when test="@data_type='FLT_1D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_2D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=2, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_3D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=3, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_4D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=4, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_5D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=5, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />", coordinate5="<xsl:value-of select="@coordinate5" />") </xsl:when>   
    <xsl:when test="@data_type='FLT_6D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=6, data_type=float, coordinate1="<xsl:value-of select="@coordinate1" />", coordinate2="<xsl:value-of select="@coordinate2" />", coordinate3="<xsl:value-of select="@coordinate3" />", coordinate4="<xsl:value-of select="@coordinate4" />", coordinate5="<xsl:value-of select="@coordinate5" />", coordinate6="<xsl:value-of select="@coordinate6" />") </xsl:when>   
    <xsl:when test="@data_type='cpx_type'">       <xsl:value-of select="@name"/>:complex    = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when>   
    <xsl:when test="@data_type='cplx_1d_type'">   <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=complex, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>   
    <xsl:when test="@data_type='CPX_0D'">         <xsl:value-of select="@name"/>:complex    = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>") </xsl:when>   
    <xsl:when test="@data_type='CPX_1D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=1, data_type=complex, coordinate1="<xsl:value-of select="@coordinate1" />") </xsl:when>   
    <xsl:when test="@data_type='CPX_2D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=2, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_3D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=3, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_4D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=4, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_5D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=5, data_type=complex) </xsl:when>   
    <xsl:when test="@data_type='CPX_6D'">         <xsl:value-of select="@name"/>:ndFunction = sp_property(type="<xsl:value-of select="@type"/>", units="<xsl:value-of select="@units"/>", ndims=6, data_type=complex) </xsl:when>   

    <xsl:when test="@data_type='structure'">      <xsl:value-of select="@name"/>:_T_<xsl:value-of select="@structure_reference"/>  = sp_property() </xsl:when>
    <xsl:when test="@data_type='struct_array'">   <xsl:value-of select="@name"/>:List[_T_<xsl:value-of select="@structure_reference"/>]   = sp_property(coordinate1="<xsl:value-of select="@coordinate1"/>") </xsl:when>
    <xsl:otherwise> # unknown data type <xsl:value-of select="@name"/>:_T_<xsl:value-of select="@structure_reference"/>  </xsl:otherwise>
  </xsl:choose>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""
</xsl:template>

<xsl:template match = "field[@doc_identifier]" mode = "DECLARE"> 
<xsl:variable name="ext_doc" select="document(concat($DD_PATH,@doc_identifier))/constants" />
<xsl:text>&#xA;    </xsl:text>
  <xsl:choose>
    <xsl:when test="@data_type='structure'"><xsl:value-of select="@name"/>:_E_<xsl:value-of select="$ext_doc/[@name]"/>  = sp_property() </xsl:when>
    <xsl:when test="@data_type='struct_array'"><xsl:value-of select="@name"/>:List[_E_<xsl:value-of select="$ext_doc/[@name]"/>]   = sp_property(coordinate1="<xsl:value-of select="@coordinate1"/>") </xsl:when>
    <xsl:otherwise> # unknown data type <xsl:value-of select="@name"/>:_E_<xsl:value-of select="@name"/>  </xsl:otherwise>
  </xsl:choose>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""
</xsl:template>
 
<xsl:template match = "field[@doc_identifier]" mode = "DEFINE"> 
  <xsl:variable name="ext_doc" select="document(concat($DD_PATH,@doc_identifier))/constants" />
class _E_<xsl:value-of select="$ext_doc/[@name]"/>(IntFlag):
    """<xsl:value-of select="my:line-wrap($ext_doc/header, $line-width, 7)"/>"""
  <xsl:for-each select="$ext_doc/int">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="@name"/>:   <xsl:value-of select="."/>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@description, $line-width, 7)"/>"""
  </xsl:for-each>
</xsl:template>

<!-- Define field dataclass -->
<xsl:template match = "field[not(@doc_identifier)]" mode = "DEFINE"> 
<xsl:variable name="cls_name" select="@structure_reference"/>
<xsl:choose>
<xsl:when test="(/IDSs/utilities/field[(@data_type='structure' and @name=$cls_name)])"> </xsl:when>
<xsl:otherwise>        

<xsl:apply-templates select="field[(@data_type='structure' or @data_type='struct_array')]" mode = "DEFINE"/>  

class _T_<xsl:value-of select="$cls_name"/>(Dict[Node]):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""

    <xsl:apply-templates select="field[substring(@name,string-length(@name)-string-length('_error_upper')+1)!='_error_upper'
      and substring(@name,string-length(@name)-string-length('_error_lower')+1)!='_error_lower'
      and substring(@name,string-length(@name)-string-length('_error_index')+1)!='_error_index'
      ]" mode = "DECLARE"/>
    
</xsl:otherwise>
</xsl:choose>
</xsl:template>


<xsl:template match = "field[(@doc_identifier)]" mode = "DEFINE_UTIL">     

<xsl:variable name="ext_doc" select="document(concat($DD_PATH,@doc_identifier))/constants" />
class _E_<xsl:value-of select="$ext_doc/[@name]"/>(IntFlag):
    """<xsl:value-of select="my:line-wrap($ext_doc/header, $line-width, 7)"/>"""
  <xsl:for-each select="$ext_doc/int">
<xsl:text>&#xA;    </xsl:text><xsl:value-of select="@name"/>:   <xsl:value-of select="."/>
<xsl:text>&#xA;    </xsl:text>"""<xsl:value-of select="my:line-wrap(@description, $line-width, 7)"/>"""
  </xsl:for-each>
    
 
</xsl:template>

<xsl:template match = "field[not(@doc_identifier)]" mode = "DEFINE_UTIL">     
<xsl:apply-templates select="field[(@data_type='structure' or @data_type='struct_array')]" mode = "DEFINE_UTIL"/>  

class _T_<xsl:value-of select="@name"/>(Dict[Node]):
    """<xsl:value-of select="my:line-wrap(@documentation, $line-width, 7)"/>"""

    <xsl:apply-templates select="field[substring(@name,string-length(@name)-string-length('_error_upper')+1)!='_error_upper'
      and substring(@name,string-length(@name)-string-length('_error_lower')+1)!='_error_lower'
      and substring(@name,string-length(@name)-string-length('_error_index')+1)!='_error_index'
      ]" mode = "DECLARE"/>
    
 
</xsl:template>

</xsl:stylesheet>
