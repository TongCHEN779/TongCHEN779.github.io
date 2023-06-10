---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

Journal Articles
---
<ol>
{% for post in site.publications reversed %}
	{% if post.pubtype == 'journal' %}
  		<li> {% include archive-single.html %} </li>
	{% endif %}
{% endfor %}
</ol>

Peer-Reviewed Conference Papers
---
<ol>
{% for post in site.publications reversed %}
	{% if post.pubtype == 'conference' %}
  		<li> {% include archive-single.html %} </li>
	{% endif %}
{% endfor %}
</ol>

Preprints / Under Review
---
<ol>
{% for post in site.publications reversed %}
	{% if post.pubtype == 'preprint' %}
  		<li> {% include archive-single.html %} </li>
	{% endif %}
{% endfor %}
</ol>

Theses
---
<ol>
{% for post in site.publications reversed %}
	{% if post.pubtype == 'thesis' %}
  		<li> {% include archive-single.html %} </li>
	{% endif %}
{% endfor %}
</ol>