<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet  
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions"	
  xmlns:my="http://www.example.com/my"  
  xmlns:saxon="http://saxon.sf.net/"
  version="3.0"
>

<xsl:function name="my:to-camel-case" as="xs:string">
  <xsl:param name="string" as="xs:string"/>
  <xsl:sequence select="string-join(tokenize($string, '_')!(upper-case(substring(., 1, 1)) || lower-case(substring(., 2))))"/>
</xsl:function>

<xsl:function name="my:line-wrap" as="xs:string">
  <xsl:param name="text" as="xs:string" />
  <xsl:param name="line-length" as="xs:integer" />
  <xsl:param name="indent" as="xs:integer" />
  <xsl:variable name="spaces" select="string-join((for $i in 1 to $indent return '&#x9;'), '')" />
  <xsl:variable name="wrapped-text" select="replace(concat(normalize-space(translate($text, '&quot;', '_')),' '), concat('(.{0,', $line-length, '}) '), concat('$1&#10;', $spaces))" />
  <xsl:sequence select="substring($wrapped-text, 1, string-length($wrapped-text) - $indent - 1)" />
</xsl:function>

<xsl:function name="my:indent">
  <xsl:param name="num"/>
  <xsl:sequence select="translate(substring('00000', 1, $num), '0', '&#x9;')"/>
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

</xsl:stylesheet>