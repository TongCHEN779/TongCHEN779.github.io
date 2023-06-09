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

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}