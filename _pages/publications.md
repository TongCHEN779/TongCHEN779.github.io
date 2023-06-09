---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
author_googlescholar: true
---

{% if author_googlescholar %}
  You can also find my articles on <u><a href="{{https://scholar.google.com/citations?user=KArfuYIAAAAJ}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

Journal Articles
---
<ol>
{% for post in site.publications reversed %}
  {% if post.type == 'journal' %}
     <li> {% include archive-single.html %} </li>
  {% endif %}
{% endfor %}
</ol>

Peer-reviewed Conference Papers
---
<ol>
{% for post in site.publications reversed %}
  {% if post.type == 'conference' %}
     <li> {% include archive-single.html %} </li>
  {% endif %}
{% endfor %}
</ol>

Preprints / Under Review
---
<ol>
{% for post in site.publications reversed %}
  {% if post.type == 'preprint' %}
     <li> {% include archive-single.html %} </li>
  {% endif %}
{% endfor %}
</ol>

Thesis
---
<ol>
{% for post in site.publications reversed %}
  {% if post.type == 'thesis' %}
     <li> {% include archive-single.html %} </li>
  {% endif %}
{% endfor %}
</ol>